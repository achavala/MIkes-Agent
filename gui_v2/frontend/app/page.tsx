"use client";
import React, { useEffect, useState } from 'react';
import { MetricCard, cn } from '@/components/MetricCard';
import { Button } from '@/components/Button';
import { Sidebar } from '@/components/Sidebar';
import { Activity, BarChart2, Shield, Settings, Wallet, PlayCircle, PauseCircle, RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import TradesPage from './trades/page';
import BacktestingPage from './backtesting/page';

export default function Home() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [portfolio, setPortfolio] = useState<any>(null);
  const [agentStatus, setAgentStatus] = useState<any>({ status: 'unknown' });
  const [positions, setPositions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [backendConnected, setBackendConnected] = useState(false);

  // Poll backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch Portfolio
        try {
          const portRes = await fetch('http://localhost:8000/api/portfolio');
          if (portRes.ok) {
            const portData = await portRes.json();
            setPortfolio(portData);
            setBackendConnected(true);
          } else {
            console.warn("Portfolio fetch failed:", portRes.status);
            setBackendConnected(false);
            // Set fallback data
            if (!portfolio) {
              setPortfolio({ equity: 0, daily_pnl_pct: 0, daily_pnl_usd: 0, buying_power: 0 });
            }
          }
        } catch (e) {
          console.error("Portfolio fetch error:", e);
          setBackendConnected(false);
          if (!portfolio) {
            setPortfolio({ equity: 0, daily_pnl_pct: 0, daily_pnl_usd: 0, buying_power: 0 });
          }
        }

        // Fetch Agent Status
        try {
          const statusRes = await fetch('http://localhost:8000/api/agent/status');
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setAgentStatus(statusData);
          } else {
            setAgentStatus({ status: 'stopped', last_heartbeat: new Date().toISOString() });
          }
        } catch (e) {
          console.error("Status fetch error:", e);
          setAgentStatus({ status: 'stopped', last_heartbeat: new Date().toISOString() });
        }

        // Fetch Positions
        try {
          const posRes = await fetch('http://localhost:8000/api/positions');
          if (posRes.ok) {
            const posData = await posRes.json();
            setPositions(posData || []);
          } else {
            setPositions([]);
          }
        } catch (e) {
          console.error("Positions fetch error:", e);
          setPositions([]);
        }
        
      } catch (err) {
        console.error("Error fetching backend data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  // --- Render Views ---

  const renderDashboard = () => (
    <>
      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard 
          label="Total Equity" 
          value={portfolio && portfolio.equity !== undefined ? `$${portfolio.equity.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '$0.00'}
          subValue={portfolio && portfolio.daily_pnl_pct !== undefined ? `${portfolio.daily_pnl_pct >= 0 ? '+' : ''}${portfolio.daily_pnl_pct.toFixed(2)}% Today` : '0.00% Today'}
          trend={portfolio?.daily_pnl_pct >= 0 ? 'up' : 'down'}
        />
        <MetricCard 
          label="Today's P&L" 
          value={portfolio && portfolio.daily_pnl_usd !== undefined ? `$${portfolio.daily_pnl_usd >= 0 ? '+' : ''}${portfolio.daily_pnl_usd.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '$0.00'}
          trend={portfolio?.daily_pnl_usd >= 0 ? 'up' : 'down'}
        />
        <MetricCard 
          label="Buying Power" 
          value={portfolio && portfolio.buying_power !== undefined ? `$${portfolio.buying_power.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '$0.00'}
        />
        <MetricCard 
          label="Agent Status" 
          value={agentStatus.status === 'running' ? 'ONLINE' : 'OFFLINE'}
          subValue={agentStatus.status === 'running' ? `Last Heartbeat: ${new Date(agentStatus.last_heartbeat).toLocaleTimeString()}` : 'Stopped'}
          className={agentStatus.status === 'running' ? 'border-emerald-500/20' : 'border-rose-500/20'}
          trend={agentStatus.status === 'running' ? 'up' : 'down'}
        />
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Chart Area */}
        <div className="lg:col-span-2 glass-card rounded-3xl p-8 min-h-[400px]">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-white">Performance Analytics</h2>
            <div className="flex gap-2">
              {['1H', '1D', '1W', '1M'].map((tf) => (
                <button key={tf} className="px-3 py-1 text-sm rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-800 rounded-xl min-h-[300px]">
            <p className="text-slate-600">Performance Chart Component (Recharts Integration)</p>
          </div>
        </div>

        {/* Active Positions */}
        <div className="glass-card rounded-3xl p-8">
          <h2 className="text-xl font-semibold text-white mb-6">Active Positions</h2>
          <div className="space-y-4">
            {positions.length === 0 ? (
                <div className="text-slate-500 text-sm text-center py-4">No active positions</div>
            ) : (
                positions.map((pos, i) => (
                  <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer border border-white/5">
                    <div>
                      <div className="font-bold text-white">{pos.symbol}</div>
                      <div className="text-xs text-slate-400">{pos.qty} shares</div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-white">${pos.market_value?.toLocaleString()}</div>
                      <div className={pos.unrealized_pl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {pos.unrealized_pl >= 0 ? '+' : ''}{pos.unrealized_plpc?.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))
            )}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <main className="min-h-screen bg-[#050511] text-slate-200">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      
      <div className="pl-64 p-8 lg:p-12 transition-all duration-300">
        {/* Header - Only show for non-Dashboard pages */}
        {activePage !== 'Dashboard' && (
          <header className="flex justify-between items-center mb-12 fade-in">
            <div>
              <h2 className="text-3xl font-bold text-white mb-1">{activePage}</h2>
              <p className="text-slate-400 text-sm">Real-time trading overview and performance</p>
            </div>
            
            <div className="flex gap-4">
              <Button variant="secondary" className="gap-2">
                <Settings className="w-4 h-4" /> Settings
              </Button>
              <Button variant="primary" glow className="gap-2">
                <PlayCircle className="w-4 h-4" /> Start Agent
              </Button>
            </div>
          </header>
        )}

        {/* Page Content */}
        <div className="fade-in">
            {activePage === 'Dashboard' && (
              <>
                {/* Top Action Bar for Dashboard */}
                <div className="flex justify-between items-center mb-8">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium",
                      backendConnected 
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" 
                        : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                    )}>
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        backendConnected ? "bg-emerald-400 shadow-[0_0_8px_#4ade80]" : "bg-rose-400"
                      )} />
                      {backendConnected ? "Backend Connected" : "Backend Disconnected"}
                    </div>
                  </div>
                  <div className="flex gap-4">
                    <Button variant="secondary" className="gap-2">
                      <Settings className="w-4 h-4" /> Settings
                    </Button>
                    <Button variant="primary" glow className="gap-2">
                      {agentStatus.status === 'running' ? (
                        <>
                          <PauseCircle className="w-4 h-4" /> Stop Agent
                        </>
                      ) : (
                        <>
                          <PlayCircle className="w-4 h-4" /> Start Agent
                        </>
                      )}
                    </Button>
                  </div>
                </div>
                {loading && !portfolio ? (
                  <div className="flex items-center justify-center min-h-[400px]">
                    <div className="text-center">
                      <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mb-4"></div>
                      <p className="text-slate-400">Connecting to backend...</p>
                    </div>
                  </div>
                ) : (
                  renderDashboard()
                )}
              </>
            )}
            {activePage === 'Trades' && (
              <div className="fade-in">
                <TradesPage />
              </div>
            )}
            {activePage === 'Backtesting' && (
              <div className="fade-in">
                <BacktestingPage />
              </div>
            )}
            {activePage !== 'Dashboard' && activePage !== 'Trades' && activePage !== 'Backtesting' && (
                <div className="glass-card rounded-3xl p-12 text-center border-dashed border-2 border-slate-800">
                    <h3 className="text-2xl font-bold text-white mb-4">Coming Soon</h3>
                    <p className="text-slate-400">The {activePage} module is currently under development.</p>
                </div>
            )}
        </div>
      </div>
    </main>
  );
}
