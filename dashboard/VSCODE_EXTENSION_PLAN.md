# VS Code Extension Porting Plan

The current React dashboard can be ported to a VS Code extension using a **Webview**.

## Architecture
1.  **Frontend**: Reuse the React app (`dashboard/`).
    -   Build the React app into a single HTML/JS bundle (using Vite).
    -   Load this bundle into a VS Code WebviewPanel.
    -   Use `acquireVsCodeApi()` for messaging instead of `fetch()`.
2.  **Backend**: Replace `server.py` with a TypeScript Extension Host logic.
    -   Instead of a Python FastAPI server, the VS Code extension itself (running in Node.js) will handle file I/O (`jobs.json`), file watching, and command execution (`start_job`, `delete_job`).
    -   Communication happens via `webview.postMessage` (Ext -> UI) and `vscode.postMessage` (UI -> Ext).

## Migration Steps
1.  **Initialize Extension**: Use `yo code` or manually setup `package.json` with `engines: { vscode: "^1.x" }`.
2.  **Webview Provider**: Create a `DashboardPanel` class in `extension.ts` that creates the webview and serves the built React assets.
3.  **Message Passing Layer**:
    -   In React: Create a `vscodeService.ts` that mocks `fetch` calls by sending messages to the extension host.
    -   In Extension: Handle messages like `GET_JOBS`, `DELETE_JOB`, `OPEN_JOB` and perform file operations directly using `vscode.workspace.fs` or Node `fs`.
4.  **Job Logger**:
    -   The Python `JobLogger` needs to remain if Python scripts are writing to it.
    -   However, the *Extension* will read `jobs.json` directly to update the UI, replacing the need for `server.py`.
    -   Or, keep `server.py` running and have the Extension just wrap the URL in a webview (simplest "iframe" approach), but a true native extension avoids the extra server process.

## Recommended Approach (Native)
Keep `job_logger.py` for Python scripts to write data.
Make the VS Code Extension read `jobs.json` and push updates to the React Webview.
Remove `server.py` entirely in this mode.

## Implementation
(This would require a new `extension/` directory and significant boilerplate setup).

