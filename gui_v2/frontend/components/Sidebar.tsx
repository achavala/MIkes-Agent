import React from 'react';
import { Home, BarChart2, Briefcase, Users, Shield, Settings, TestTube } from 'lucide-react';
import { cn } from './MetricCard';

interface SidebarProps {
  activePage: string;
  onNavigate: (page: string) => void;
}

export function Sidebar({ activePage, onNavigate }: SidebarProps) {
  const menuItems = [
    { name: 'Dashboard', icon: Home },
    { name: 'Analytics', icon: BarChart2 },
    { name: 'Trades', icon: Briefcase },
    { name: 'Backtesting', icon: TestTube },
    { name: 'Agents', icon: Users },
    { name: 'Risk Management', icon: Shield },
    { name: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 h-screen fixed left-0 top-0 glass-card border-r border-white/5 flex flex-col p-6 z-50 bg-[#050511]/80 backdrop-blur-xl">
      <div className="mb-10 px-2">
        <h1 className="text-2xl font-bold">
          <span className="text-white">Mike</span>
          <span className="text-gradient">Agent</span>
        </h1>
        <p className="text-slate-500 text-xs mt-1 uppercase tracking-wider">Pro Terminal</p>
      </div>

      <nav className="flex-1 space-y-2">
        {menuItems.map((item) => {
          const isActive = activePage === item.name;
          return (
            <button
              key={item.name}
              onClick={() => onNavigate(item.name)}
              className={cn(
                "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group",
                isActive 
                  ? "bg-primary/10 text-white shadow-[0_0_20px_rgba(99,102,241,0.3)] border border-primary/20" 
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              )}
            >
              <item.icon className={cn("w-5 h-5", isActive ? "text-primary" : "group-hover:text-white")} />
              <span className="font-medium">{item.name}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto pt-6 border-t border-white/5">
        <div className="flex items-center gap-3 px-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_#4ade80]" />
          <span className="text-sm text-slate-400">System Operational</span>
        </div>
      </div>
    </aside>
  );
}

