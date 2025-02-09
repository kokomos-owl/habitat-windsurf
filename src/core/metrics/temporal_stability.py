"""Enhanced temporal stability calculation for metric flows."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

@dataclass
class TimeWindow:
    """Represents a temporal window for stability calculation."""
    start_time: datetime
    end_time: datetime
    weight: float = 1.0
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_metric(self, metric_type: str, value: float) -> None:
        """Add a metric value to the window."""
        self.metrics[metric_type].append(value)
    
    def get_stability(self, metric_type: str) -> float:
        """Calculate stability score for a metric type."""
        if metric_type not in self.metrics:
            return 0.0
            
        values = self.metrics[metric_type]
        if not values:
            return 0.0
            
        # Calculate statistical measures
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0
        cv = std / mean if mean != 0 else 0
        
        # Higher stability for lower coefficient of variation
        return max(0.0, min(1.0, 1.0 - cv))

class TemporalStabilityTracker:
    """Tracks temporal stability of metrics."""
    
    def __init__(self, window_size: timedelta = timedelta(days=30)):
        self.window_size = window_size
        self.windows: List[TimeWindow] = []
        self.current_window: Optional[TimeWindow] = None
        
        # Stability thresholds
        self.stability_threshold = 0.7
        self.high_stability_threshold = 0.9
        
        # Trend detection settings
        self.trend_threshold = 0.1
        self.trend_window_count = 3
        
    def add_observation(self, timestamp: datetime, metric_type: str, 
                       value: float, context: Optional[Dict] = None) -> None:
        """Add a new metric observation."""
        # Create or update current window
        if not self.current_window or timestamp > self.current_window.end_time:
            self._create_new_window(timestamp)
            
        self.current_window.add_metric(metric_type, value)
        
    def get_stability_score(self, metric_type: str) -> float:
        """Calculate overall stability score for a metric type."""
        if not self.windows:
            return 0.0
            
        # Calculate weighted stability across windows
        total_weight = 0.0
        weighted_stability = 0.0
        
        for window in self.windows:
            stability = window.get_stability(metric_type)
            weighted_stability += stability * window.weight
            total_weight += window.weight
            
        return weighted_stability / total_weight if total_weight > 0 else 0.0
    
    def detect_trends(self, metric_type: str) -> Dict[str, Any]:
        """Detect trends in metric stability."""
        if len(self.windows) < self.trend_window_count:
            return {'trend': 'insufficient_data'}
            
        recent_windows = self.windows[-self.trend_window_count:]
        stabilities = [w.get_stability(metric_type) for w in recent_windows]
        
        # Calculate trend
        trend = np.polyfit(range(len(stabilities)), stabilities, 1)[0]
        
        result = {
            'trend': 'stable',
            'confidence': min(1.0, max(0.0, abs(trend) * 5))  # Scale trend to confidence
        }
        
        if trend > self.trend_threshold:
            result['trend'] = 'improving'
        elif trend < -self.trend_threshold:
            result['trend'] = 'degrading'
            
        return result
    
    def get_stability_report(self, metric_type: str) -> Dict[str, Any]:
        """Generate comprehensive stability report."""
        stability_score = self.get_stability_score(metric_type)
        trends = self.detect_trends(metric_type)
        
        return {
            'stability_score': stability_score,
            'trend': trends['trend'],
            'trend_confidence': trends.get('confidence', 0.0),
            'window_count': len(self.windows),
            'current_window_metrics': len(self.current_window.metrics.get(metric_type, [])) if self.current_window else 0,
            'assessment': self._assess_stability(stability_score)
        }
    
    def _create_new_window(self, timestamp: datetime) -> None:
        """Create a new time window."""
        end_time = timestamp + self.window_size
        
        # Create new window
        new_window = TimeWindow(
            start_time=timestamp,
            end_time=end_time,
            weight=1.0  # New windows start with full weight
        )
        
        # Age existing windows
        self._age_windows()
        
        # Add new window
        self.windows.append(new_window)
        self.current_window = new_window
        
        # Prune old windows
        self._prune_old_windows()
    
    def _age_windows(self) -> None:
        """Age existing windows by reducing their weights."""
        for window in self.windows:
            # Exponential decay of window weights
            window.weight *= 0.9
    
    def _prune_old_windows(self) -> None:
        """Remove windows with negligible weight."""
        self.windows = [w for w in self.windows if w.weight > 0.1]
    
    def _assess_stability(self, stability_score: float) -> str:
        """Assess stability level."""
        if stability_score >= self.high_stability_threshold:
            return 'highly_stable'
        elif stability_score >= self.stability_threshold:
            return 'stable'
        elif stability_score >= self.stability_threshold * 0.7:
            return 'moderately_stable'
        else:
            return 'unstable'
