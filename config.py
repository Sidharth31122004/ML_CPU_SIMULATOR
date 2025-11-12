"""
Configuration management for the scheduling optimizer
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class SchedulingConfig:
    """Configuration for scheduling simulations"""
    
    # Dataset configuration
    num_processes: int = 100
    arrival_distribution: str = "exponential"
    burst_distribution: str = "lognormal"
    priority_range: tuple = (1, 10)
    seed: Optional[int] = None
    
    # Algorithm configuration
    time_quantum: float = 4.0
    enable_aging: bool = True
    max_priority_boost: int = 5
    
    # ML model configuration
    test_split: float = 0.2
    random_state: int = 42
    
    # UI configuration
    show_gantt_chart: bool = True
    show_comparative_analysis: bool = True
    export_csv: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'num_processes': self.num_processes,
            'arrival_distribution': self.arrival_distribution,
            'burst_distribution': self.burst_distribution,
            'time_quantum': self.time_quantum,
        }
