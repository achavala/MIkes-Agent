import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ label, value, subValue, trend, className }: MetricCardProps) {
  return (
    <div className={cn(
      "glass-card p-6 rounded-2xl transition-all duration-300 hover:translate-y-[-2px] hover:border-primary/30 hover:shadow-[0_8px_32px_rgba(99,102,241,0.2)]",
      className
    )}>
      <h3 className="text-slate-400 text-xs font-semibold tracking-wider uppercase mb-3">{label}</h3>
      <div className="text-3xl font-bold text-gradient mb-2 leading-tight">{value}</div>
      {subValue && (
        <div className={cn(
          "text-sm font-medium flex items-center gap-1",
          trend === 'up' ? "text-emerald-400" : trend === 'down' ? "text-rose-400" : "text-slate-500"
        )}>
          {trend === 'up' && <span className="text-emerald-400">↑</span>}
          {trend === 'down' && <span className="text-rose-400">↓</span>}
          {subValue}
        </div>
      )}
    </div>
  );
}

