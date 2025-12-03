import React, { useState, useEffect, useRef } from 'react';
import { Job, JobStatus, JobType, AnalysisResult } from '../types';
import { Terminal, Activity, Cpu, Database, AlertTriangle, Wand2, Play, XCircle, Clock, Server, Code, Trash2, FolderOpen, Wifi, WifiOff } from 'lucide-react';
import { analyzeJobFailure } from '../services/geminiService';
import { deleteJob, openJobInEditor, createLogStreamWebSocket, getJobLogs } from '../services/jobService';

interface JobDetailProps {
  job: Job | null;
  onClose: () => void;
}

const JobDetail: React.FC<JobDetailProps> = ({ job, onClose }) => {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isOpening, setIsOpening] = useState(false);
  const [streamedLogs, setStreamedLogs] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!job) return;

    if (job.status === JobStatus.RUNNING) {
      setIsStreaming(true);
      setStreamError(false);

      const ws = createLogStreamWebSocket(job.id);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected for job', job.id);
      };

      ws.onmessage = (event) => {
        setStreamedLogs((prev) => prev + event.data);
      };

      ws.onerror = () => {
        setStreamError(true);
        setIsStreaming(false);
      };

      ws.onclose = () => {
        setIsStreaming(false);
      };

      return () => {
        ws.close();
      };
    } else {
      getJobLogs(job.id).then((logs) => {
        setStreamedLogs(logs);
      });
    }
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamedLogs]);

  if (!job) {
    return (
      <div className="h-full flex items-center justify-center text-vscode-text opacity-50">
        <div className="text-center">
          <Activity className="w-16 h-16 mx-auto mb-4" />
          <p>Select a job to view details</p>
        </div>
      </div>
    );
  }

  const handleDelete = async () => {
    if (!window.confirm(`Are you sure you want to delete job "${job.name}"?`)) return;
    
    setIsDeleting(true);
    try {
        await deleteJob(job.id);
        onClose(); // Close detail view, main list should refresh
    } catch (error) {
        console.error("Failed to delete job:", error);
        alert("Failed to delete job. See console.");
    } finally {
        setIsDeleting(false);
    }
  };

  const handleOpenInEditor = async () => {
    setIsOpening(true);
    try {
        await openJobInEditor(job.id);
    } catch (error) {
        console.error("Failed to open job:", error);
        alert(`Failed to open job folder: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
        setIsOpening(false);
    }
  };

  const handleAnalyze = async () => {
    setAnalysis({ isLoading: true, summary: '', possibleCause: '', suggestedFix: '' });
    try {
      const result = await analyzeJobFailure(job);
      setAnalysis({
        isLoading: false,
        summary: result.summary,
        possibleCause: result.rootCause,
        suggestedFix: result.recommendation
      });
    } catch (err) {
        console.error(err);
      setAnalysis({
        isLoading: false,
        summary: '',
        possibleCause: '',
        suggestedFix: '',
        error: "Failed to analyze log. Check API Key."
      });
    }
  };

  const getStatusColor = (status: JobStatus) => {
    switch (status) {
      case JobStatus.RUNNING: return 'text-vscode-blue';
      case JobStatus.COMPLETED: return 'text-vscode-green';
      case JobStatus.FAILED: return 'text-vscode-red';
      case JobStatus.PENDING: return 'text-vscode-orange';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="h-full flex flex-col bg-vscode-sidebar border-l border-vscode-activity transition-colors">
      {/* Header */}
      <div className="p-6 border-b border-vscode-activity flex justify-between items-start">
        <div>
          <h2 className="text-xl font-semibold text-vscode-header flex items-center gap-2">
            {job.name}
            <span className={`text-xs border px-2 py-0.5 rounded-full ${getStatusColor(job.status)} border-current opacity-80`}>
              {job.status}
            </span>
          </h2>
          <div className="text-sm text-vscode-text mt-1 flex items-center gap-4">
            <span className="flex items-center gap-1"><Clock size={14} /> {new Date(job.startTime).toLocaleTimeString()}</span>
            <span className="flex items-center gap-1"><Server size={14} /> {job.type}</span>
            {job.slurm && <span className="flex items-center gap-1">#{job.slurm.jobId}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
             <button 
                onClick={handleDelete}
                disabled={isDeleting}
                className="p-2 text-gray-500 hover:text-vscode-red transition-colors rounded hover:bg-vscode-activity"
                title="Delete Job"
             >
                 <Trash2 size={18} />
             </button>
            <button onClick={onClose} className="p-2 text-vscode-text hover:text-vscode-header transition-colors rounded hover:bg-vscode-activity">
                <XCircle size={20} />
            </button>
        </div>
      </div>

      {/* Content Scrollable */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        
        {/* Configuration Section (Hydra or Generic) */}
        <div className="bg-vscode-bg rounded-lg p-4 border border-vscode-activity relative group">
             <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <button 
                    onClick={handleOpenInEditor}
                    disabled={isOpening}
                    className="flex items-center gap-1.5 text-xs bg-vscode-activity hover:bg-vscode-accent hover:text-white px-2 py-1 rounded transition-colors text-vscode-text"
                    title="Open Output Directory in VS Code"
                >
                    <FolderOpen size={14} /> Open in Editor
                </button>
             </div>

            <h3 className="text-sm font-semibold text-vscode-blue mb-3 flex items-center gap-2">
                {job.hydra ? <Database size={16}/> : <Code size={16} />} 
                {job.hydra ? 'Hydra Configuration' : 'Job Details'}
            </h3>
            
            {job.hydra ? (
                <>
                    <div className="grid grid-cols-2 gap-4 text-sm text-vscode-text">
                        <div>
                            <span className="block text-xs text-gray-500">Config Name</span>
                            {job.hydra.configName}
                        </div>
                        <div>
                            <span className="block text-xs text-gray-500">Output Dir</span>
                            <span className="font-mono text-xs break-all">{job.hydra.outputDir}</span>
                        </div>
                    </div>
                    {job.hydra.overrides.length > 0 && (
                        <div className="mt-3">
                             <span className="block text-xs text-gray-500 mb-1">Overrides</span>
                             <div className="flex flex-wrap gap-2">
                                {job.hydra.overrides.map((ov, i) => (
                                    <span key={i} className="bg-vscode-activity px-2 py-1 rounded text-xs font-mono text-vscode-blue">
                                        {ov}
                                    </span>
                                ))}
                             </div>
                        </div>
                    )}
                </>
            ) : (
                <div className="text-sm text-vscode-text">
                     {job.command && (
                        <div className="mb-2">
                            <span className="block text-xs text-gray-500">Command</span>
                            <code className="bg-vscode-activity px-2 py-1 rounded text-xs font-mono block mt-1">{job.command}</code>
                        </div>
                     )}
                     <div className="grid grid-cols-2 gap-4">
                         <div>
                            <span className="block text-xs text-gray-500">Job ID</span>
                            <span className="font-mono text-xs">{job.id}</span>
                         </div>
                     </div>
                </div>
            )}
        </div>

        {/* Resources (If Running/Slurm) */}
        {(job.status === JobStatus.RUNNING || job.type === JobType.SLURM) && (
             <div className="grid grid-cols-2 gap-4">
                 <div className="bg-vscode-bg p-4 rounded-lg border border-vscode-activity">
                    <h4 className="text-gray-500 text-xs mb-1 flex items-center gap-1"><Cpu size={14}/> CPU / Node</h4>
                    <div className="text-lg font-mono text-vscode-green">
                        {job.type === JobType.SLURM ? job.slurm?.nodeList : `${job.cpuUsage?.toFixed(2)}%`}
                    </div>
                 </div>
                 <div className="bg-vscode-bg p-4 rounded-lg border border-vscode-activity">
                    <h4 className="text-gray-500 text-xs mb-1 flex items-center gap-1"><Activity size={14}/> Memory / Partition</h4>
                    <div className="text-lg font-mono text-vscode-blue">
                        {job.type === JobType.SLURM ? job.slurm?.partition : `${job.memoryUsage?.toFixed(2)} MB`}
                    </div>
                 </div>
             </div>
        )}

        {/* Logs & Analysis */}
        <div className="flex flex-col gap-4">
            <div className="flex justify-between items-center">
                <h3 className="text-sm font-semibold text-vscode-header flex items-center gap-2">
                    <Terminal size={16}/> Logs
                    {isStreaming && (
                      <span className="flex items-center gap-1 text-xs text-vscode-green">
                        <Wifi size={12} className="animate-pulse" /> Live
                      </span>
                    )}
                    {streamError && (
                      <span className="flex items-center gap-1 text-xs text-vscode-red">
                        <WifiOff size={12} /> Stream Error
                      </span>
                    )}
                </h3>
                {job.status === JobStatus.FAILED && (
                    <button
                        onClick={handleAnalyze}
                        disabled={analysis?.isLoading}
                        className="flex items-center gap-2 px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs rounded transition-colors disabled:opacity-50"
                    >
                        <Wand2 size={14} />
                        {analysis?.isLoading ? 'Thinking...' : 'Analyze Failure'}
                    </button>
                )}
            </div>

            {/* Gemini Analysis Result */}
            {analysis && (
                <div className="bg-indigo-900/20 border border-indigo-500/30 p-4 rounded-lg animate-in fade-in slide-in-from-top-4">
                    {analysis.error ? (
                        <p className="text-vscode-red text-sm">{analysis.error}</p>
                    ) : analysis.isLoading ? (
                        <div className="flex items-center gap-3 text-indigo-500 dark:text-indigo-300 text-sm">
                             <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                             Analyzing logs with Gemini 2.5 Flash...
                        </div>
                    ) : (
                        <div className="space-y-3 text-sm">
                             <div className="flex gap-2">
                                <span className="font-bold text-indigo-600 dark:text-indigo-300 min-w-[80px]">Summary:</span>
                                <span className="text-gray-800 dark:text-gray-200">{analysis.summary}</span>
                             </div>
                             <div className="flex gap-2">
                                <span className="font-bold text-vscode-red min-w-[80px]">Cause:</span>
                                <span className="text-gray-800 dark:text-gray-200">{analysis.possibleCause}</span>
                             </div>
                             <div className="flex gap-2">
                                <span className="font-bold text-vscode-green min-w-[80px]">Fix:</span>
                                <span className="text-gray-800 dark:text-gray-200">{analysis.suggestedFix}</span>
                             </div>
                        </div>
                    )}
                </div>
            )}

            <div className="bg-gray-100 dark:bg-black rounded border border-vscode-activity p-3 font-mono text-xs h-96 overflow-y-auto text-gray-800 dark:text-gray-300 whitespace-pre-wrap transition-colors relative">
                {streamedLogs || <span className="text-gray-500 italic">No log output yet.</span>}
                <div ref={logEndRef} />
            </div>
        </div>

      </div>
    </div>
  );
};

export default JobDetail;