import asyncio
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from job_logger import JobLogger
from job_scanner import JobScanner
from log_watcher import LogWatcher
from slurm_monitor import SlurmMonitor

logger = JobLogger()
scanner = JobScanner(logger)
log_watcher = LogWatcher(logger)
slurm_monitor = SlurmMonitor(logger)


async def periodic_scan():
    while True:
        await asyncio.sleep(30)
        try:
            scanner.sync_jobs()
            if slurm_monitor.slurm_available:
                slurm_monitor.update_slurm_jobs()
        except Exception as e:
            print(f"Error during periodic scan: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting initial job scan...")
    stats = scanner.sync_jobs()
    print(f"Initial scan complete: {stats}")

    task = asyncio.create_task(periodic_scan())

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/jobs")
async def get_jobs():
    """
    Get all jobs from the JSON store.
    """
    return logger.get_jobs()

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job by ID. Also deletes the output directory if it exists.
    """
    job = logger.get_job(job_id)
    if not job:
         raise HTTPException(status_code=404, detail="Job not found")

    # Attempt to delete directory
    if job.get("hydra") and job["hydra"].get("outputDir"):
        output_dir = Path(job["hydra"]["outputDir"])
        if output_dir.exists() and output_dir.is_dir():
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Failed to remove directory {output_dir}: {e}")
                # We proceed to remove the job entry even if dir removal fails,
                # but maybe we should warn? For now, log and proceed.

    success = logger.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found in registry")
        
    return {"status": "deleted", "id": job_id}

@app.post("/api/jobs/{job_id}/open")
async def open_job_in_editor(job_id: str):
    """
    Open the job's output directory in VS Code.
    """
    job = logger.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Determine path to open
    path_to_open = None
    if job.get("hydra") and job["hydra"].get("outputDir"):
        path_to_open = job["hydra"]["outputDir"]
    
    if not path_to_open:
        raise HTTPException(status_code=400, detail="Job has no output directory recorded")
    
    if not os.path.exists(path_to_open):
        raise HTTPException(status_code=404, detail=f"Output directory not found: {path_to_open}")

    try:
        # Use 'code' command which should be available if using VS Code / Cursor
        if shutil.which("code"):
            # Try to open a file inside the dir to avoid switching workspace context
            # Prefer run.log, then config.yaml, then the dir itself
            log_file = os.path.join(path_to_open, "run.log")
            config_file = os.path.join(path_to_open, ".hydra", "config.yaml")
            
            target = path_to_open
            if os.path.exists(log_file):
                target = log_file
            elif os.path.exists(config_file):
                target = config_file
                
            # Use -r to reuse the window
            subprocess.Popen(["code", "-r", target])
            return {"status": "opened", "path": target}
        else:
             raise Exception("'code' command not found in PATH")
            
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to open editor: {str(e)}")


@app.post("/api/jobs/clear")
async def clear_jobs():
    """
    Clear all jobs (for testing/resetting).
    """
    logger.clear_all_jobs()
    return {"status": "cleared"}


@app.post("/api/jobs/scan")
async def scan_jobs():
    """
    Manually trigger a job scan.
    """
    stats = scanner.sync_jobs()
    if slurm_monitor.slurm_available:
        slurm_stats = slurm_monitor.update_slurm_jobs()
        stats["slurm"] = slurm_stats
    return {"status": "scanned", "stats": stats}


@app.get("/api/jobs/{job_id}/children")
async def get_job_children(job_id: str):
    """
    Get child jobs of a sweep parent job.
    """
    jobs = logger.get_jobs()
    children = [j for j in jobs if j.get("parent_id") == job_id]
    children.sort(key=lambda j: j.get("sweep_index", 0))
    return children


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, lines: int | None = None):
    """
    Get logs for a job. Optionally tail last N lines.
    """
    job = logger.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if lines is not None:
        log_content = log_watcher.tail_log(job_id, n_lines=lines)
    else:
        log_content = log_watcher.get_full_log(job_id)

    return {"id": job_id, "logs": log_content}


@app.websocket("/api/jobs/{job_id}/logs/stream")
async def stream_job_logs(websocket: WebSocket, job_id: str):
    """
    Stream logs for a job in real-time via WebSocket.
    """
    await websocket.accept()

    job = logger.get_job(job_id)
    if not job:
        await websocket.send_text("Error: Job not found")
        await websocket.close()
        return

    try:
        async for log_chunk in log_watcher.watch_log(job_id, poll_interval=1.0):
            await websocket.send_text(log_chunk)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Error: {e}")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
