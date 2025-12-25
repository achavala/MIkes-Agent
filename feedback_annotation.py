"""
FEEDBACK ANNOTATION SYSTEM
Ensures feedback is used for analysis, not direct retraining
"""
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json


class FeedbackAnnotationSystem:
    """
    Feedback annotation system that prevents direct retraining from feedback
    
    Rules:
    - Feedback annotates logs, doesn't directly retrain
    - Feedback used as analysis labels, regime notes, future supervised targets
    - NOT used as immediate reward shaping
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.annotations: Dict[str, List[Dict]] = {}
    
    def annotate_decision(
        self,
        timestamp: datetime,
        symbol: str,
        annotation_type: str,  # "GOOD", "CONCERNING", "INCORRECT"
        comment: str,
        reviewer: str = "PM",
        severity: str = "MEDIUM"
    ):
        """
        Annotate a specific decision (does NOT retrain)
        
        Args:
            timestamp: Decision timestamp
            symbol: Trading symbol
            annotation_type: GOOD, CONCERNING, INCORRECT
            comment: Annotation comment
            reviewer: Reviewer name
            severity: LOW, MEDIUM, HIGH
        """
        annotation = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "annotation_type": annotation_type,
            "comment": comment,
            "reviewer": reviewer,
            "severity": severity,
            "created_at": datetime.now().isoformat(),
            "used_for_retraining": False  # Explicit flag
        }
        
        date_key = timestamp.strftime("%Y-%m-%d")
        if date_key not in self.annotations:
            self.annotations[date_key] = []
        
        self.annotations[date_key].append(annotation)
        
        # Save to file
        self._save_annotations()
    
    def get_annotations_for_analysis(
        self,
        start_date: str,
        end_date: str,
        annotation_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get annotations for analysis (NOT for direct retraining)
        
        Returns annotations that can be used as:
        - Analysis labels
        - Regime notes
        - Future supervised learning targets
        """
        annotations = []
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        for date_key, date_annotations in self.annotations.items():
            date = datetime.fromisoformat(date_key)
            if start <= date <= end:
                for ann in date_annotations:
                    if annotation_type is None or ann['annotation_type'] == annotation_type:
                        annotations.append(ann)
        
        return annotations
    
    def get_regime_notes(self, regime: str) -> List[Dict]:
        """Get annotations for specific regime (for regime-specific analysis)"""
        notes = []
        for date_annotations in self.annotations.values():
            for ann in date_annotations:
                if regime.lower() in ann.get('comment', '').lower():
                    notes.append(ann)
        return notes
    
    def _save_annotations(self):
        """Save annotations to file"""
        annotations_file = self.log_dir / "annotations.json"
        try:
            with open(annotations_file, 'w') as f:
                json.dump(self.annotations, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save annotations: {e}")
    
    def load_annotations(self):
        """Load annotations from file"""
        annotations_file = self.log_dir / "annotations.json"
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r') as f:
                    self.annotations = json.load(f)
            except Exception:
                self.annotations = {}


# Global instance
_annotation_system: Optional[FeedbackAnnotationSystem] = None


def initialize_annotation_system(log_dir: str = "logs") -> FeedbackAnnotationSystem:
    """Initialize global annotation system"""
    global _annotation_system
    _annotation_system = FeedbackAnnotationSystem(log_dir=log_dir)
    _annotation_system.load_annotations()
    return _annotation_system


def get_annotation_system() -> Optional[FeedbackAnnotationSystem]:
    """Get global annotation system instance"""
    return _annotation_system





