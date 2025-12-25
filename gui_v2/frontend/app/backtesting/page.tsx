"use client";
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/Button';
import { MetricCard, cn } from '@/components/MetricCard';
import { Calendar, Play, TrendingUp, TrendingDown, DollarSign, BarChart3, Clock } from 'lucide-react';

export default function Backtesting() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [symbols, setSymbols] = useState('SPY,QQQ');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<any>(null);

  // Load available date range
  useEffect(() => {
    const fetchDateRange = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/backtest/available-dates');
        if (res.ok) {
          const data = await res.json();
          setDateRange(data);
          // Set default dates (30 days back to today)
          const today = new Date();
          const thirtyDaysAgo = new Date(today);
          thirtyDaysAgo.setDate(today.getDate() - 30);
          
          setEndDate(today.toISOString().split('T')[0]);
          setStartDate(thirtyDaysAgo.toISOString().split('T')[0]);
        }
      } catch (e) {
        console.error("Error fetching date range:", e);
      }
    };
    fetchDateRange();
  }, []);

  const handleRunBacktest = async () => {
    if (!startDate || !endDate) {
      setError("Please select start and end dates");
      return;
    }

    if (new Date(startDate) > new Date(endDate)) {
      setError("Start date must be before end date");
      return;
    }

    setRunning(true);
    setError(null);
    setResults(null);

    try {
      const res = await fetch('http://localhost:8000/api/backtest/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          symbols: symbols.split(',').map(s => s.trim()),
          initial_capital: initialCapital,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Backtest failed');
      }

      const data = await res.json();
      setResults(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run backtest');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Configuration Card */}
      <div className="glass-card rounded-3xl p-8">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
          <Calendar className="w-6 h-6 text-primary" />
          Backtest Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              max={endDate || dateRange?.latest_date}
              min={dateRange?.earliest_date}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              max={dateRange?.latest_date}
              min={startDate || dateRange?.earliest_date}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Symbols (comma-separated)</label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="SPY,QQQ,SPX"
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Initial Capital ($)</label>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(parseFloat(e.target.value) || 10000)}
              min={100}
              step={100}
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            />
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400">
            {error}
          </div>
        )}

        <Button
          onClick={handleRunBacktest}
          disabled={running}
          variant="primary"
          glow
          className="w-full md:w-auto gap-2"
        >
          {running ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
              Running Backtest...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" /> Run Backtest
            </>
          )}
        </Button>
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              label="Total P&L"
              value={`$${results.total_pnl?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || '0.00'}`}
              subValue={`${results.return_pct >= 0 ? '+' : ''}${results.return_pct?.toFixed(2) || '0.00'}% Return`}
              trend={results.total_pnl >= 0 ? 'up' : 'down'}
              className={results.total_pnl >= 0 ? 'border-emerald-500/20' : 'border-rose-500/20'}
            />
            <MetricCard
              label="Final Capital"
              value={`$${results.final_capital?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || '0.00'}`}
              subValue={`From $${initialCapital.toLocaleString()}`}
            />
            <MetricCard
              label="Total Trades"
              value={results.total_trades || 0}
              subValue={`Win Rate: ${results.win_rate?.toFixed(1) || '0.0'}%`}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${results.max_drawdown?.toFixed(2) || '0.00'}%`}
              trend={results.max_drawdown > -10 ? 'up' : 'down'}
            />
          </div>

          {/* Detailed Results Card */}
          <div className="glass-card rounded-3xl p-8">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              Backtest Results
            </h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-slate-400 mb-1">Period</div>
                  <div className="text-white font-semibold">
                    {startDate} to {endDate}
                  </div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Symbols</div>
                  <div className="text-white font-semibold">{symbols}</div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Initial Capital</div>
                  <div className="text-white font-semibold">${initialCapital.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-slate-400 mb-1">Final Capital</div>
                  <div className={cn(
                    "font-semibold",
                    results.final_capital >= initialCapital ? "text-emerald-400" : "text-rose-400"
                  )}>
                    ${results.final_capital?.toLocaleString()}
                  </div>
                </div>
              </div>

              {results.trades && results.trades.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-white mb-4">Trade History</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left py-3 px-4 text-slate-400">Date</th>
                          <th className="text-left py-3 px-4 text-slate-400">Symbol</th>
                          <th className="text-left py-3 px-4 text-slate-400">Action</th>
                          <th className="text-right py-3 px-4 text-slate-400">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.trades.slice(0, 20).map((trade: any, i: number) => (
                          <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                            <td className="py-3 px-4 text-slate-300">{trade.date || 'N/A'}</td>
                            <td className="py-3 px-4 text-white font-medium">{trade.symbol || 'N/A'}</td>
                            <td className="py-3 px-4 text-slate-300">{trade.action || 'N/A'}</td>
                            <td className={cn(
                              "py-3 px-4 text-right font-semibold",
                              trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}>
                              ${trade.pnl?.toFixed(2) || '0.00'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}





