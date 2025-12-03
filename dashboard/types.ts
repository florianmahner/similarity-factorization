export enum JobStatus {
  RUNNING = 'RUNNING',
  PENDING = 'PENDING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
}

export enum JobType {
  LOCAL = 'LOCAL',
  SLURM = 'SLURM',
}

export interface HydraConfig {
  configName: string;
  overrides: string[];
  outputDir: string;
}

export interface Job {
  id: string;
  name: string;
  type: JobType;
  status: JobStatus;
  startTime: string;
  duration?: string;
  cpuUsage?: number; // Percentage
  memoryUsage?: number; // MB
  hydra?: HydraConfig; // Made optional for generic jobs
  command?: string; // For non-hydra jobs
  slurm?: {
    partition?: string;
    nodeList?: string;
    jobId?: string;
    jobName?: string;
    priority?: number;
  };
  screen?: {
    id: string;
    pid: string;
    name: string;
    state: string;
  };
  logs: {
    stdout: string;
    stderr: string;
  };
  parent_id?: string;
  sweep_index?: number;
  is_sweep?: boolean;
  sweep_progress?: {
    total: number;
    completed: number;
    failed: number;
    running: number;
  };
}

export interface AnalysisResult {
  summary: string;
  possibleCause: string;
  suggestedFix: string;
  isLoading: boolean;
  error?: string;
}