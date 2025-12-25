"""
LOG COMPRESSION AND INDEXING
Handles log volume management for 30-day backtest
"""
import gzip
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def compress_daily_logs(log_dir: str = "logs", category: str = None):
    """
    Compress daily log files (gzip)
    
    Args:
        log_dir: Log directory
        category: Specific category to compress (None = all)
    """
    log_path = Path(log_dir)
    
    categories = [category] if category else ["decisions", "risk", "execution", "positions", "learning", "feedback"]
    
    for cat in categories:
        cat_dir = log_path / cat
        if not cat_dir.exists():
            continue
        
        # Find all JSONL files
        for log_file in cat_dir.glob("*.jsonl"):
            # Skip if already compressed
            if log_file.name.endswith('.gz'):
                continue
            
            # Compress
            compressed_file = log_path / cat / f"{log_file.stem}.jsonl.gz"
            
            try:
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove original (keep compressed)
                log_file.unlink()
                
                print(f"✅ Compressed: {log_file.name} → {compressed_file.name}")
            except Exception as e:
                print(f"⚠️ Error compressing {log_file.name}: {e}")


def load_compressed_log(log_file: Path) -> List[Dict]:
    """Load compressed log file"""
    try:
        with gzip.open(log_file, 'rt') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading compressed log: {e}")
        return []


def get_log_metadata(log_dir: str = "logs") -> Dict:
    """Get log metadata (counts per day)"""
    log_path = Path(log_dir)
    metadata = {}
    
    categories = ["decisions", "risk", "execution", "positions", "learning", "feedback"]
    
    for cat in categories:
        cat_dir = log_path / cat
        if not cat_dir.exists():
            continue
        
        metadata[cat] = {}
        
        # Check index file first
        index_file = log_path / f"{cat}_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index = json.load(f)
                    metadata[cat] = index
                    continue
            except Exception:
                pass
        
        # Fallback: count files
        for log_file in cat_dir.glob("*.jsonl*"):
            date = log_file.stem.replace('.jsonl', '')
            
            if date not in metadata[cat]:
                metadata[cat][date] = {'count': 0}
            
            # Count lines (approximate for compressed)
            try:
                if log_file.name.endswith('.gz'):
                    with gzip.open(log_file, 'rt') as f:
                        count = sum(1 for _ in f)
                else:
                    with open(log_file, 'r') as f:
                        count = sum(1 for _ in f)
                
                metadata[cat][date]['count'] = count
            except Exception:
                pass
    
    return metadata





