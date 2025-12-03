import React, { useState, useEffect, useMemo } from 'react';
import { Job, JobStatus, JobType } from './types';
import { getJobs, getJobChildren, triggerJobScan } from './services/jobService';
import JobDetail from './components/JobDetail';
import {
  LayoutDashboard,
  Server,
  Laptop,
  Settings,
  RefreshCw,
  Search,
  Filter,
  Play,
  AlertCircle,
  Sun,
  Moon,
  X,
  Calendar,
  Clock,
  ChevronRight,
  ChevronDown
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'local' | 'slurm'>('dashboard');
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [timeFilter, setTimeFilter] = useState<'24h' | '7d' | '30d' | 'all'>('7d');
  const [isLoading, setIsLoading] = useState(true);
  const [connectionError, setConnectionError] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [refreshRate, setRefreshRate] = useState(5000);
  const [expandedSweeps, setExpandedSweeps] = useState<Set<string>>(new Set());
  const [sweepChildren, setSweepChildren] = useState<Record<string, Job[]>>({});

  // Theme Toggle Logic
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'dark' | 'light' | null;
    if (savedTheme) {
        setTheme(savedTheme);
        if (savedTheme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    } else {
        document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleTheme = () => {
      const newTheme = theme === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
      localStorage.setItem('theme', newTheme);
      if (newTheme === 'dark') {
          document.documentElement.classList.add('dark');
      } else {
          document.documentElement.classList.remove('dark');
      }
  };

  const fetchData = async () => {
    setIsLoading(true);
    const data = await getJobs();
    setJobs(data);
    if (data.length > 0 && data[0].id.startsWith('mock-')) {
        setConnectionError(true);
    } else {
        setConnectionError(false);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchData();
    
    if (refreshRate === 0) return; // Manual refresh mode

    const interval = setInterval(() => {
        getJobs().then(data => {
             setJobs(data);
             if (data.length > 0 && !data[0].id.startsWith('mock-')) {
                setConnectionError(false);
             }
        });
    }, refreshRate);
    return () => clearInterval(interval);
  }, [refreshRate]);

  const toggleSweepExpansion = async (jobId: string) => {
    const newExpanded = new Set(expandedSweeps);
    if (newExpanded.has(jobId)) {
      newExpanded.delete(jobId);
      setExpandedSweeps(newExpanded);
    } else {
      newExpanded.add(jobId);
      setExpandedSweeps(newExpanded);

      if (!sweepChildren[jobId]) {
        const children = await getJobChildren(jobId);
        setSweepChildren(prev => ({ ...prev, [jobId]: children }));
      }
    }
  };

  const filteredJobs = useMemo(() => {
    const now = new Date();
    return jobs.filter(job => {
      if (job.parent_id) return false;

      const matchesSearch = job.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                            job.id.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesTab = activeTab === 'dashboard' ? true :
                         activeTab === 'local' ? job.type === JobType.LOCAL :
                         job.type === JobType.SLURM;

      const jobTime = new Date(job.startTime);
      let matchesTime = true;
      if (timeFilter === '24h') {
        matchesTime = (now.getTime() - jobTime.getTime()) < 24 * 60 * 60 * 1000;
      } else if (timeFilter === '7d') {
        matchesTime = (now.getTime() - jobTime.getTime()) < 7 * 24 * 60 * 60 * 1000;
      } else if (timeFilter === '30d') {
        matchesTime = (now.getTime() - jobTime.getTime()) < 30 * 24 * 60 * 60 * 1000;
      }

      return matchesSearch && matchesTab && matchesTime;
    });
  }, [jobs, activeTab, searchQuery, timeFilter]);

  const stats = useMemo(() => {
    const now = new Date();
    const timeFilteredOnly = jobs.filter(job => {
        const jobTime = new Date(job.startTime);
        if (timeFilter === '24h') return (now.getTime() - jobTime.getTime()) < 24 * 60 * 60 * 1000;
        if (timeFilter === '7d') return (now.getTime() - jobTime.getTime()) < 7 * 24 * 60 * 60 * 1000;
        if (timeFilter === '30d') return (now.getTime() - jobTime.getTime()) < 30 * 24 * 60 * 60 * 1000;
        return true;
    });

    return {
      running: timeFilteredOnly.filter(j => j.status === JobStatus.RUNNING).length,
      failed: timeFilteredOnly.filter(j => j.status === JobStatus.FAILED).length,
      pending: timeFilteredOnly.filter(j => j.status === JobStatus.PENDING).length,
      completed: timeFilteredOnly.filter(j => j.status === JobStatus.COMPLETED).length,
      totalLocal: timeFilteredOnly.filter(j => j.type === JobType.LOCAL).length,
      totalSlurm: timeFilteredOnly.filter(j => j.type === JobType.SLURM).length,
    };
  }, [jobs, timeFilter]);

  const chartData = [
    { name: 'Running', value: stats.running, color: '#4fc1ff' },
    { name: 'Failed', value: stats.failed, color: '#f14c4c' },
    { name: 'Pending', value: stats.pending, color: '#cca700' },
    { name: 'Completed', value: stats.completed, color: '#6a9955' },
  ];

  const renderSettingsModal = () => {
    if (!isSettingsOpen) return null;
    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center backdrop-blur-sm" onClick={() => setIsSettingsOpen(false)}>
            <div className="bg-vscode-bg border border-vscode-activity rounded-lg shadow-xl w-96 overflow-hidden animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
                <div className="p-4 border-b border-vscode-activity flex justify-between items-center bg-vscode-sidebar">
                    <h3 className="text-vscode-header font-medium">Settings</h3>
                    <button onClick={() => setIsSettingsOpen(false)} className="text-gray-500 hover:text-vscode-text transition-colors">
                        <X size={18} />
                    </button>
                </div>
                <div className="p-4 space-y-5">
                    {/* Theme */}
                    <div>
                        <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider font-semibold">Appearance</label>
                        <div className="flex gap-2 bg-vscode-item p-1 rounded-md border border-vscode-activity">
                            <button 
                                onClick={() => theme !== 'light' && toggleTheme()}
                                className={`flex-1 py-1.5 text-xs rounded-sm flex items-center justify-center gap-2 transition-all ${theme === 'light' ? 'bg-vscode-bg text-vscode-header shadow-sm' : 'text-gray-500 hover:text-vscode-text'}`}
                            >
                                <Sun size={14} /> Light
                            </button>
                            <button 
                                onClick={() => theme !== 'dark' && toggleTheme()}
                                className={`flex-1 py-1.5 text-xs rounded-sm flex items-center justify-center gap-2 transition-all ${theme === 'dark' ? 'bg-vscode-bg text-vscode-header shadow-sm' : 'text-gray-500 hover:text-vscode-text'}`}
                            >
                                <Moon size={14} /> Dark
                            </button>
                        </div>
                    </div>

                     {/* Refresh Rate */}
                    <div>
                        <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider font-semibold">Refresh Interval</label>
                        <select 
                            value={refreshRate} 
                            onChange={(e) => setRefreshRate(Number(e.target.value))}
                            className="w-full bg-vscode-item border border-vscode-activity text-vscode-text text-sm rounded p-2 focus:outline-none focus:border-vscode-accent appearance-none cursor-pointer"
                        >
                            <option value={2000}>Fast (2s)</option>
                            <option value={5000}>Normal (5s)</option>
                            <option value={10000}>Slow (10s)</option>
                            <option value={0}>Manual (Off)</option>
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Controls how often job status is updated.</p>
                    </div>
                    
                    {/* About */}
                    <div>
                        <label className="block text-xs text-gray-500 mb-2 uppercase tracking-wider font-semibold">System Status</label>
                        <div className="text-xs text-vscode-text p-3 bg-vscode-item rounded border border-vscode-activity space-y-2">
                            <div className="flex justify-between">
                                <span>Observer Version</span>
                                <span className="font-mono opacity-80">v1.0.0</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Backend Status</span>
                                <span className={`flex items-center gap-1.5 font-medium ${connectionError ? 'text-vscode-red' : 'text-vscode-green'}`}>
                                    <div className={`w-2 h-2 rounded-full ${connectionError ? 'bg-vscode-red' : 'bg-vscode-green'}`}></div>
                                    {connectionError ? 'Disconnected' : 'Active'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="p-3 border-t border-vscode-activity bg-vscode-sidebar flex justify-end">
                    <button 
                        onClick={() => setIsSettingsOpen(false)}
                        className="px-4 py-1.5 bg-vscode-accent text-white text-xs rounded hover:opacity-90 transition-opacity"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    )
  }

  const renderJobRow = (job: Job, isChild: boolean = false) => {
    const isExpanded = expandedSweeps.has(job.id);
    const isSweep = job.is_sweep;
    const children = sweepChildren[job.id] || [];

    return (
      <div key={job.id}>
        <div
          className={`
            group flex items-center justify-between p-3 rounded-md border border-transparent
            ${isChild ? 'ml-8 bg-vscode-item' : ''}
            ${selectedJob?.id === job.id ? 'bg-vscode-activity border-vscode-accent' : 'hover:bg-vscode-hover'}
          `}
        >
          <div className="flex items-center gap-4 flex-1">
            {isSweep && !isChild && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleSweepExpansion(job.id);
                }}
                className="text-gray-500 hover:text-vscode-text"
              >
                {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              </button>
            )}
            <div
              className="flex items-center gap-4 flex-1 cursor-pointer"
              onClick={() => setSelectedJob(job)}
            >
              <div className={`w-2 h-2 rounded-full ${
                  job.status === JobStatus.RUNNING ? 'bg-vscode-blue shadow-[0_0_8px_rgba(79,193,255,0.6)]' :
                  job.status === JobStatus.FAILED ? 'bg-vscode-red' :
                  job.status === JobStatus.PENDING ? 'bg-vscode-orange' : 'bg-vscode-green'
              }`} />
              <div className="flex-1">
                <div className="font-medium text-sm text-vscode-header">
                  {job.name}
                  {isSweep && job.sweep_progress && (
                    <span className="ml-2 text-xs text-gray-500 font-normal">
                      ({job.sweep_progress.completed}/{job.sweep_progress.total} completed)
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500 flex gap-2 mt-0.5 font-mono">
                  <span>{job.id.substring(0, 8)}</span>
                  {job.type === JobType.SLURM && <span className="text-vscode-text">• {job.slurm?.partition || 'SLURM'}</span>}
                  {job.screen && <span className="text-vscode-blue">• Screen: {job.screen.name}</span>}
                  {job.hydra ? (
                      <span className="text-vscode-text">• {job.hydra.overrides.length} overrides</span>
                  ) : (
                      <span className="text-vscode-text">• Generic Job</span>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className={`text-xs font-medium ${
                     job.status === JobStatus.RUNNING ? 'text-vscode-blue' :
                     job.status === JobStatus.FAILED ? 'text-vscode-red' :
                     job.status === JobStatus.PENDING ? 'text-vscode-orange' : 'text-vscode-green'
                }`}>{job.status}</div>
                <div className="text-xs text-gray-500 mt-0.5">
                    {new Date(job.startTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </div>
              </div>
            </div>
          </div>
        </div>

        {isSweep && isExpanded && children.length > 0 && (
          <div className="mt-1 space-y-1">
            {children.map(child => renderJobRow(child, true))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-vscode-bg text-vscode-text font-sans overflow-hidden selection:bg-vscode-accent selection:text-white">
      
      {/* Modals */}
      {renderSettingsModal()}

      {/* Sidebar */}
      <div className="w-16 bg-vscode-sidebar flex flex-col items-center py-4 border-r border-vscode-activity z-20">
        <div className="mb-6">
             <button 
                onClick={() => setActiveTab('dashboard')}
                className="text-vscode-blue hover:scale-110 transition-transform"
                title="Home"
             >
                <Server size={24} strokeWidth={1.5} />
             </button>
        </div>
        <nav className="flex flex-col gap-4 w-full">
            <button 
                onClick={() => setActiveTab('dashboard')}
                className={`p-3 w-full flex justify-center border-l-2 transition-all ${activeTab === 'dashboard' ? 'border-vscode-accent text-vscode-header bg-vscode-bg' : 'border-transparent text-gray-500 hover:text-vscode-text'}`}
                title="Dashboard"
            >
                <LayoutDashboard size={20} strokeWidth={1.5} />
            </button>
            <button 
                onClick={() => setActiveTab('local')}
                className={`p-3 w-full flex justify-center border-l-2 transition-all ${activeTab === 'local' ? 'border-vscode-accent text-vscode-header bg-vscode-bg' : 'border-transparent text-gray-500 hover:text-vscode-text'}`}
                title="Local Jobs"
            >
                <Laptop size={20} strokeWidth={1.5} />
            </button>
            <button 
                onClick={() => setActiveTab('slurm')}
                className={`p-3 w-full flex justify-center border-l-2 transition-all ${activeTab === 'slurm' ? 'border-vscode-accent text-vscode-header bg-vscode-bg' : 'border-transparent text-gray-500 hover:text-vscode-text'}`}
                title="Slurm Jobs"
            >
                <Server size={20} strokeWidth={1.5} />
            </button>
        </nav>
        <div className="mt-auto mb-4">
            <button 
                onClick={() => setIsSettingsOpen(true)}
                className={`p-3 hover:text-vscode-text transition-colors ${isSettingsOpen ? 'text-vscode-accent' : 'text-gray-500'}`}
                title="Settings"
            >
                <Settings size={20} strokeWidth={1.5} />
            </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        
        {/* Top Bar */}
        <header className="h-12 border-b border-vscode-activity flex items-center justify-between px-4 bg-vscode-bg transition-colors">
            <h1 className="font-semibold text-sm text-vscode-header tracking-wide flex items-center gap-2">
                <span className="opacity-50 uppercase tracking-wider text-xs">Project /</span> 
                DASHBOARD
                {connectionError && (
                    <span className="flex items-center gap-1 ml-4 text-xs text-vscode-orange bg-vscode-orange/10 px-2 py-0.5 rounded">
                        <AlertCircle size={12} /> Observer Disconnected
                    </span>
                )}
            </h1>
            <div className="flex items-center gap-3">
                <div className="relative group">
                    <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-vscode-accent" />
                    <input 
                        type="text" 
                        placeholder="Search jobs..." 
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="bg-vscode-activity rounded-md py-1 pl-8 pr-3 text-xs text-vscode-header focus:outline-none focus:ring-1 focus:ring-vscode-accent w-48 focus:w-64 transition-all border border-transparent placeholder-gray-500"
                    />
                </div>

                <div className="h-4 w-px bg-vscode-activity mx-1"></div>

                {/* Time Filter */}
                <div className="relative group flex items-center">
                     <div className="absolute left-2 text-gray-500 pointer-events-none">
                        <Calendar size={14} />
                     </div>
                     <select 
                        value={timeFilter}
                        onChange={(e) => setTimeFilter(e.target.value as any)}
                        className="bg-vscode-activity text-vscode-text text-xs rounded-md py-1 pl-8 pr-2 border border-transparent focus:border-vscode-accent appearance-none cursor-pointer hover:text-vscode-header transition-colors focus:outline-none"
                     >
                        <option value="24h">Last 24 Hours</option>
                        <option value="7d">Last 7 Days</option>
                        <option value="30d">Last 30 Days</option>
                        <option value="all">All Time</option>
                     </select>
                </div>
                
                <div className="h-4 w-px bg-vscode-activity mx-1"></div>
                
                {/* Theme Toggle (Quick Access) */}
                <button
                    onClick={toggleTheme}
                    className="p-1.5 hover:bg-vscode-activity rounded text-gray-400 hover:text-vscode-header transition-colors"
                    title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
                >
                    {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                </button>

                <button 
                    onClick={() => fetchData()} 
                    className={`p-1.5 hover:bg-vscode-activity rounded text-gray-400 hover:text-vscode-header transition-colors ${isLoading ? 'animate-spin' : ''}`}
                    title="Refresh"
                >
                    <RefreshCw size={14} />
                </button>
            </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 flex overflow-hidden">
            
            {/* Job List / Dashboard Area */}
            <div className={`flex-1 overflow-y-auto p-6 ${selectedJob ? 'hidden lg:block lg:w-1/2 xl:w-3/5' : 'w-full'}`}>
                
                {activeTab === 'dashboard' && (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                         {/* Overview Cards */}
                         <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-vscode-item p-4 rounded-lg border border-vscode-activity hover:border-vscode-accent transition-colors group">
                                <div className="text-xs text-gray-500 mb-1 group-hover:text-vscode-blue transition-colors flex justify-between">
                                    <span>Active Jobs</span>
                                    <ActivityIcon status="running" />
                                </div>
                                <div className="text-2xl font-light text-vscode-header">{stats.running}</div>
                            </div>
                             <div className="bg-vscode-item p-4 rounded-lg border border-vscode-activity hover:border-vscode-red transition-colors group">
                                <div className="text-xs text-gray-500 mb-1 group-hover:text-vscode-red transition-colors flex justify-between">
                                    <span>Failures ({timeFilter})</span>
                                    <ActivityIcon status="failed" />
                                </div>
                                <div className="text-2xl font-light text-vscode-header">{stats.failed}</div>
                            </div>
                             <div className="bg-vscode-item p-4 rounded-lg border border-vscode-activity hover:border-vscode-orange transition-colors group">
                                <div className="text-xs text-gray-500 mb-1 group-hover:text-vscode-orange transition-colors flex justify-between">
                                    <span>Queue</span>
                                    <Clock size={14} />
                                </div>
                                <div className="text-2xl font-light text-vscode-header">{stats.pending}</div>
                            </div>
                         </div>

                         {/* Stats Chart */}
                         <div className="bg-vscode-item p-4 rounded-lg border border-vscode-activity h-64">
                            <h3 className="text-sm font-medium text-vscode-text mb-4 flex justify-between">
                                <span>Job Distribution</span>
                                <span className="text-xs text-gray-500 font-normal bg-vscode-bg px-2 py-0.5 rounded border border-vscode-activity">{timeFilter} view</span>
                            </h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={theme === 'dark' ? '#444' : '#e5e5e5'} />
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="name" type="category" width={80} tick={{fill: theme === 'dark' ? '#ccc' : '#666', fontSize: 12}} tickLine={false} axisLine={false} />
                                    <Tooltip 
                                        contentStyle={{backgroundColor: theme === 'dark' ? '#252526' : '#fff', borderColor: theme === 'dark' ? '#333' : '#e5e5e5', color: theme === 'dark' ? '#fff' : '#000'}} 
                                        itemStyle={{color: theme === 'dark' ? '#fff' : '#000'}}
                                        cursor={{fill: theme === 'dark' ? '#ffffff10' : '#00000005'}}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                         </div>
                    </div>
                )}

                {/* Job List Header */}
                <div className="mt-6 mb-4 flex justify-between items-end">
                    <h2 className="text-lg font-light text-vscode-header">
                        {activeTab === 'dashboard' ? 'Recent Activity' : `${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Jobs`}
                    </h2>
                    <div className="text-xs text-gray-500">
                        Showing {filteredJobs.length} jobs ({timeFilter})
                    </div>
                </div>

                <div className="space-y-2">
                    {connectionError && (
                        <div className="bg-vscode-orange/20 border border-vscode-orange/50 text-vscode-orange text-xs p-3 rounded mb-4 flex items-center gap-2">
                            <AlertCircle size={16} />
                            <span>
                                <strong>Observer Offline:</strong> Run <code>./dashboard</code> in your project root. Showing demo data.
                            </span>
                        </div>
                    )}
                    {filteredJobs.length === 0 ? (
                         <div className="text-center py-12 text-gray-500 border-2 border-dashed border-vscode-activity rounded-lg">
                            No jobs found matching your criteria.
                         </div>
                    ) : (
                        filteredJobs.map(renderJobRow)
                    )}
                </div>

            </div>

            {/* Detail Pane */}
            {(selectedJob) && (
                 <div className={`
                    fixed inset-y-0 right-0 w-full lg:static lg:w-1/2 xl:w-2/5 shadow-2xl lg:shadow-none z-30 lg:z-0 transform transition-transform duration-300 ease-in-out
                    ${selectedJob ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
                 `}>
                    <JobDetail job={selectedJob} onClose={() => setSelectedJob(null)} />
                 </div>
            )}
        </main>
      </div>
    </div>
  );
};

// Helper for stats icons
const ActivityIcon = ({status}: {status: 'running' | 'failed'}) => {
    return (
        <div className={`w-2 h-2 rounded-full ${status === 'running' ? 'bg-vscode-blue' : 'bg-vscode-red'}`}></div>
    )
}

export default App;