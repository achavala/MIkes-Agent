"""
ONLINE LEARNING / DAILY RETRAINING SYSTEM
Implements daily retraining, regime-dependent retraining, rolling windows, model versioning, A/B testing
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import shutil


class ModelVersion:
    """Represents a model version"""
    def __init__(
        self,
        version_id: str,
        model_path: str,
        training_date: datetime,
        regime: str,
        performance_metrics: Dict,
        training_config: Dict
    ):
        self.version_id = version_id
        self.model_path = model_path
        self.training_date = training_date
        self.regime = regime
        self.performance_metrics = performance_metrics
        self.training_config = training_config
        self.is_active = False
        self.is_production = False
    
    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'model_path': self.model_path,
            'training_date': self.training_date.isoformat(),
            'regime': self.regime,
            'performance_metrics': self.performance_metrics,
            'training_config': self.training_config,
            'is_active': self.is_active,
            'is_production': self.is_production
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        version = cls(
            version_id=data['version_id'],
            model_path=data['model_path'],
            training_date=datetime.fromisoformat(data['training_date']),
            regime=data['regime'],
            performance_metrics=data['performance_metrics'],
            training_config=data['training_config']
        )
        version.is_active = data.get('is_active', False)
        version.is_production = data.get('is_production', False)
        return version


class OnlineLearningSystem:
    """
    Online learning system with daily retraining, regime detection, and A/B testing
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        rolling_window_days: int = 30,
        min_retrain_interval_hours: int = 20  # Retrain at least once per day
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.rolling_window_days = rolling_window_days
        self.min_retrain_interval_hours = min_retrain_interval_hours
        
        self.versions: Dict[str, ModelVersion] = {}
        self.version_history: List[ModelVersion] = []
        self.current_production_version: Optional[str] = None
        self.current_test_version: Optional[str] = None
        
        self.retrain_history: List[Dict] = []
        self.last_retrain_time: Optional[datetime] = None
        
        # Load existing versions
        self._load_version_registry()
    
    def _load_version_registry(self):
        """Load version registry from disk"""
        registry_path = self.model_dir / "version_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    for version_data in data.get('versions', []):
                        version = ModelVersion.from_dict(version_data)
                        self.versions[version.version_id] = version
                        if version.is_production:
                            self.current_production_version = version.version_id
                        if version.is_active and not version.is_production:
                            self.current_test_version = version.version_id
                    self.last_retrain_time = datetime.fromisoformat(data['last_retrain_time']) if data.get('last_retrain_time') else None
            except Exception as e:
                print(f"Warning: Could not load version registry: {e}")
    
    def _save_version_registry(self):
        """Save version registry to disk"""
        registry_path = self.model_dir / "version_registry.json"
        data = {
            'versions': [v.to_dict() for v in self.versions.values()],
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'current_production': self.current_production_version,
            'current_test': self.current_test_version
        }
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def should_retrain(self, current_regime: str) -> Tuple[bool, str]:
        """
        Determine if model should be retrained
        
        Returns:
            (should_retrain, reason)
        """
        # Check time interval
        if self.last_retrain_time:
            hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain < self.min_retrain_interval_hours:
                return False, f"Too soon since last retrain ({hours_since_retrain:.1f}h < {self.min_retrain_interval_hours}h)"
        
        # Check regime change
        if self.current_production_version:
            production_version = self.versions[self.current_production_version]
            if production_version.regime != current_regime:
                return True, f"Regime changed: {production_version.regime} → {current_regime}"
        
        # Daily retrain (if enough time has passed)
        if not self.last_retrain_time:
            return True, "First retrain"
        
        hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        if hours_since_retrain >= self.min_retrain_interval_hours:
            return True, f"Daily retrain interval reached ({hours_since_retrain:.1f}h >= {self.min_retrain_interval_hours}h)"
        
        return False, "No retrain needed"
    
    def retrain_model(
        self,
        training_data: pd.DataFrame,
        current_regime: str,
        training_config: Optional[Dict] = None
    ) -> str:
        """
        Retrain model with new data
        
        Args:
            training_data: Training data (rolling window)
            current_regime: Current market regime
            training_config: Training configuration
            
        Returns:
            New version ID
        """
        # Generate version ID
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_regime}"
        
        # Save model (placeholder - actual training would happen here)
        model_path = self.model_dir / f"{version_id}.zip"
        
        # For now, create a placeholder file
        # In production, this would train the actual RL model
        with open(model_path, 'w') as f:
            f.write(f"Model version {version_id}\nTrained on {len(training_data)} samples\nRegime: {current_regime}")
        
        # Calculate performance metrics (placeholder)
        performance_metrics = {
            'training_samples': len(training_data),
            'regime': current_regime,
            'training_date': datetime.now().isoformat(),
            'sharpe_ratio': np.random.uniform(1.0, 3.0),  # Placeholder
            'win_rate': np.random.uniform(0.7, 0.9),  # Placeholder
            'avg_return': np.random.uniform(0.05, 0.15)  # Placeholder
        }
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            model_path=str(model_path),
            training_date=datetime.now(),
            regime=current_regime,
            performance_metrics=performance_metrics,
            training_config=training_config or {}
        )
        
        # Add to registry
        self.versions[version_id] = version
        self.version_history.append(version)
        
        # Set as test version (for A/B testing)
        if self.current_test_version:
            # Deactivate old test version
            old_test = self.versions.get(self.current_test_version)
            if old_test:
                old_test.is_active = False
        
        version.is_active = True
        self.current_test_version = version_id
        
        # Update retrain history
        self.retrain_history.append({
            'version_id': version_id,
            'regime': current_regime,
            'training_samples': len(training_data),
            'timestamp': datetime.now().isoformat()
        })
        
        self.last_retrain_time = datetime.now()
        self._save_version_registry()
        
        return version_id
    
    def compare_versions(
        self,
        version_a_id: str,
        version_b_id: str,
        test_period_days: int = 7
    ) -> Dict:
        """
        Compare two model versions
        
        Returns:
            Comparison results
        """
        version_a = self.versions.get(version_a_id)
        version_b = self.versions.get(version_b_id)
        
        if not version_a or not version_b:
            return {'error': 'Version not found'}
        
        # Compare performance metrics
        comparison = {
            'version_a': version_a_id,
            'version_b': version_b_id,
            'version_a_metrics': version_a.performance_metrics,
            'version_b_metrics': version_b.performance_metrics,
            'winner': None,
            'improvement': {}
        }
        
        # Determine winner based on Sharpe ratio
        sharpe_a = version_a.performance_metrics.get('sharpe_ratio', 0)
        sharpe_b = version_b.performance_metrics.get('sharpe_ratio', 0)
        
        if sharpe_b > sharpe_a:
            comparison['winner'] = version_b_id
            comparison['improvement'] = {
                'sharpe_ratio': (sharpe_b - sharpe_a) / sharpe_a * 100,
                'win_rate': (version_b.performance_metrics.get('win_rate', 0) - 
                           version_a.performance_metrics.get('win_rate', 0)) * 100
            }
        else:
            comparison['winner'] = version_a_id
        
        return comparison
    
    def promote_to_production(self, version_id: str) -> bool:
        """
        Promote a test version to production
        
        Returns:
            Success status
        """
        version = self.versions.get(version_id)
        if not version:
            return False
        
        # Deactivate current production version
        if self.current_production_version:
            old_prod = self.versions.get(self.current_production_version)
            if old_prod:
                old_prod.is_production = False
                old_prod.is_active = False
        
        # Promote new version
        version.is_production = True
        version.is_active = True
        self.current_production_version = version_id
        
        # Deactivate test version if it was the promoted one
        if self.current_test_version == version_id:
            self.current_test_version = None
        
        self._save_version_registry()
        return True
    
    def get_rolling_window_data(
        self,
        all_data: pd.DataFrame,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get rolling window of data for training
        
        Args:
            all_data: All available data
            end_date: End date for window (default: now)
            
        Returns:
            Rolling window DataFrame
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=self.rolling_window_days)
        
        # Filter data by date
        if 'date' in all_data.columns:
            mask = (all_data['date'] >= start_date) & (all_data['date'] <= end_date)
            return all_data[mask].copy()
        elif all_data.index.name == 'date' or isinstance(all_data.index, pd.DatetimeIndex):
            mask = (all_data.index >= start_date) & (all_data.index <= end_date)
            return all_data[mask].copy()
        else:
            # Assume last N rows
            return all_data.tail(self.rolling_window_days * 20).copy()  # ~20 bars per day
    
    def get_retrain_schedule(self) -> Dict:
        """Get retraining schedule information"""
        return {
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'next_retrain_earliest': (
                (self.last_retrain_time + timedelta(hours=self.min_retrain_interval_hours)).isoformat()
                if self.last_retrain_time else datetime.now().isoformat()
            ),
            'rolling_window_days': self.rolling_window_days,
            'min_retrain_interval_hours': self.min_retrain_interval_hours,
            'total_versions': len(self.versions),
            'production_version': self.current_production_version,
            'test_version': self.current_test_version
        }


# Global instance
_online_learning_system: Optional[OnlineLearningSystem] = None


def initialize_online_learning(
    model_dir: str = "models",
    rolling_window_days: int = 30,
    min_retrain_interval_hours: int = 20
) -> OnlineLearningSystem:
    """Initialize global online learning system"""
    global _online_learning_system
    _online_learning_system = OnlineLearningSystem(
        model_dir=model_dir,
        rolling_window_days=rolling_window_days,
        min_retrain_interval_hours=min_retrain_interval_hours
    )
    return _online_learning_system


def get_online_learning_system() -> Optional[OnlineLearningSystem]:
    """Get global online learning system instance"""
    return _online_learning_system





