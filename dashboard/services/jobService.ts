import { Job, JobStatus, JobType } from "../types";

// The Python server defined in server.py runs on port 8080
const API_URL = 'http://localhost:8080/api/jobs';

export const getJobs = async (): Promise<Job[]> => {
  try {
    const response = await fetch(API_URL);
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    
    // Transform the raw data to ensure it matches our Enums
    return data.map((job: any) => ({
        ...job,
        status: job.status as JobStatus,
        type: job.type as JobType,
    }));

  } catch (error) {
    console.warn("Could not fetch from Python observer. Run './dashboard' to start the backend.");
    // Return empty list or a specific error state job if you prefer
    // For now, returning empty lets the UI show the "Observer Offline" banner handled in App.tsx
    return [];
  }
};

export const deleteJob = async (jobId: string): Promise<void> => {
    const response = await fetch(`${API_URL}/${jobId}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error('Failed to delete job');
    }
};

export const openJobInEditor = async (jobId: string): Promise<void> => {
    const response = await fetch(`${API_URL}/${jobId}/open`, {
        method: 'POST',
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to open job in editor');
    }
};

export const triggerJobScan = async (): Promise<any> => {
    const response = await fetch(`${API_URL}/scan`, {
        method: 'POST',
    });
    if (!response.ok) {
        throw new Error('Failed to trigger job scan');
    }
    return response.json();
};

export const getJobChildren = async (jobId: string): Promise<Job[]> => {
    try {
        const response = await fetch(`${API_URL}/${jobId}/children`);
        if (!response.ok) {
            throw new Error('Failed to fetch job children');
        }
        const data = await response.json();
        return data.map((job: any) => ({
            ...job,
            status: job.status as JobStatus,
            type: job.type as JobType,
        }));
    } catch (error) {
        console.error('Error fetching job children:', error);
        return [];
    }
};

export const getJobLogs = async (jobId: string, lines?: number): Promise<string> => {
    try {
        const url = lines
            ? `${API_URL}/${jobId}/logs?lines=${lines}`
            : `${API_URL}/${jobId}/logs`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch job logs');
        }
        const data = await response.json();
        return data.logs;
    } catch (error) {
        console.error('Error fetching job logs:', error);
        return 'Error loading logs';
    }
};

export const createLogStreamWebSocket = (jobId: string): WebSocket => {
    const wsUrl = `ws://localhost:8080/api/jobs/${jobId}/logs/stream`;
    return new WebSocket(wsUrl);
};

// Deprecated mock function, kept for testing reference if needed
export const getMockJobs = (): Job[] => [];
