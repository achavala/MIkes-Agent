"use client";
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/Button';
import { cn } from '@/components/MetricCard';
import { Clock, TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

export default function Trades() {
  const [trades, setTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, open, closed

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/trades/recent?limit=100');
        if (res.ok) {
          const data = await res.json();
          setTrades(data || []);
        }
      } catch (e) {
        console.error("Error fetching trades:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const filteredTrades = trades.filter(trade => {
    if (filter === 'all') return true;
    // You can add status filtering logic here
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header with Filters */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Trade History</h2>
          <p className="text-slate-400 text-sm">Complete trade history and performance</p>
        </div>
        
        <div className="flex gap-2">
          {['all', 'open', 'closed'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                filter === f
                  ? "bg-primary/20 text-white border border-primary/30"
                  : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white"
              )}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Trades Table */}
      <div className="glass-card rounded-3xl p-8">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
          </div>
        ) : filteredTrades.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-500">No trades found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 px-4 text-sm font-semibold text-slate-400">Timestamp</th>
                  <th className="text-left py-4 px-4 text-sm font-semibold text-slate-400">Symbol</th>
                  <th className="text-left py-4 px-4 text-sm font-semibold text-slate-400">Action</th>
                  <th className="text-right py-4 px-4 text-sm font-semibold text-slate-400">Quantity</th>
                  <th className="text-right py-4 px-4 text-sm font-semibold text-slate-400">Entry</th>
                  <th className="text-right py-4 px-4 text-sm font-semibold text-slate-400">Exit</th>
                  <th className="text-right py-4 px-4 text-sm font-semibold text-slate-400">P&L</th>
                  <th className="text-right py-4 px-4 text-sm font-semibold text-slate-400">P&L %</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrades.map((trade, i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="py-4 px-4 text-slate-300 text-sm">
                      {trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'N/A'}
                    </td>
                    <td className="py-4 px-4 text-white font-medium">{trade.symbol || 'N/A'}</td>
                    <td className="py-4 px-4">
                      <span className={cn(
                        "px-2 py-1 rounded text-xs font-medium",
                        trade.action === 'BUY' || trade.action === 'CALL' 
                          ? "bg-emerald-500/20 text-emerald-400"
                          : "bg-rose-500/20 text-rose-400"
                      )}>
                        {trade.action || 'N/A'}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-right text-slate-300">{trade.qty || 'N/A'}</td>
                    <td className="py-4 px-4 text-right text-slate-300">
                      {trade.entry_premium ? `$${parseFloat(trade.entry_premium).toFixed(2)}` : 'N/A'}
                    </td>
                    <td className="py-4 px-4 text-right text-slate-300">
                      {trade.exit_premium ? `$${parseFloat(trade.exit_premium).toFixed(2)}` : 'N/A'}
                    </td>
                    <td className={cn(
                      "py-4 px-4 text-right font-semibold",
                      trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}>
                      {trade.pnl !== undefined ? `$${parseFloat(trade.pnl).toFixed(2)}` : 'N/A'}
                    </td>
                    <td className={cn(
                      "py-4 px-4 text-right font-semibold",
                      trade.pnl_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}>
                      {trade.pnl_pct !== undefined ? `${trade.pnl_pct >= 0 ? '+' : ''}${parseFloat(trade.pnl_pct).toFixed(2)}%` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}





