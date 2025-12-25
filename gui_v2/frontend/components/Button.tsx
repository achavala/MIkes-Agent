import React from 'react';
import { cn } from './MetricCard';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline';
  glow?: boolean;
}

export function Button({ className, variant = 'primary', glow = true, children, ...props }: ButtonProps) {
  const baseStyles = "relative inline-flex items-center justify-center px-6 py-2.5 rounded-full font-semibold transition-all duration-300 disabled:opacity-50 disabled:pointer-events-none";
  
  const variants = {
    primary: "bg-black text-white border border-transparent bg-origin-border hover:scale-[1.02]",
    secondary: "bg-surface hover:bg-white/5 text-white border border-white/10",
    outline: "border border-primary/50 text-primary hover:bg-primary/10"
  };

  const gradientBorder = variant === 'primary' 
    ? "before:absolute before:inset-0 before:-z-10 before:p-[1px] before:rounded-full before:bg-gradient-to-r before:from-primary before:to-secondary before:content-['']" 
    : "";

  const glowEffect = glow && variant === 'primary'
    ? "hover:shadow-[0_0_20px_rgba(99,102,241,0.3)]"
    : "";

  return (
    <button 
      className={cn(baseStyles, variants[variant], gradientBorder, glowEffect, className)}
      {...props}
    >
      <span className="relative z-10">{children}</span>
    </button>
  );
}





