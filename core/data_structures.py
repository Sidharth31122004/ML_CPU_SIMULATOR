"""
Core data structures for CPU scheduling optimization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class SchedulingAlgorithm(Enum):
    """Supported scheduling algorithms"""
    FCFS = "FCFS"
    SJF = "SJF"
    ROUND_ROBIN = "Round Robin"
    PREEMPTIVE_SJF = "Preemptive SJF"
    PRIORITY = "Priority"
    PREEMPTIVE_PRIORITY = "Preemptive Priority"
    AGING = "Aging"
    MULTI_LEVEL_FEEDBACK_QUEUE = "Multi-Level Feedback Queue"


@dataclass
class Process:
    """
    Represents a single process with scheduling attributes
    """
    pid: int
    arrival_time: float
    burst_time: float
    original_burst: float
    priority: int = 5
    prev_burst: float = 0.0
    time_quantum: float = 4.0
    cpu_cores: int = 1
    io_wait_time: float = 0.0
    
    # Runtime tracking
    completion_time: float = 0.0
    waiting_time: float = 0.0
    response_time: float = -1.0
    turnaround_time: float = 0.0
    context_switches: int = 0
    cpu_utilization: float = 0.0
    aging_priority: int = field(default_factory=lambda: 5)
    
    def __post_init__(self):
        """Validate process attributes"""
        if self.burst_time <= 0:
            raise ValueError(f"Process {self.pid}: burst_time must be positive")
        if self.arrival_time < 0:
            raise ValueError(f"Process {self.pid}: arrival_time cannot be negative")
        if not 1 <= self.priority <= 10:
            raise ValueError(f"Process {self.pid}: priority must be 1-10")
    
    def reset_runtime_values(self):
        """Reset runtime tracking for re-simulation"""
        self.completion_time = 0.0
        self.waiting_time = 0.0
        self.response_time = -1.0
        self.turnaround_time = 0.0
        self.context_switches = 0
        self.cpu_utilization = 0.0
        self.aging_priority = self.priority
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'pid': self.pid,
            'arrival_time': self.arrival_time,
            'burst_time': self.burst_time,
            'priority': self.priority,
            'prev_burst': self.prev_burst,
            'time_quantum': self.time_quantum,
            'completion_time': self.completion_time,
            'waiting_time': self.waiting_time,
            'response_time': self.response_time,
            'turnaround_time': self.turnaround_time,
        }


@dataclass
class AlgorithmMetrics:
    """
    Comprehensive metrics for scheduling algorithm performance
    """
    algorithm: SchedulingAlgorithm
    avg_waiting_time: float = 0.0
    avg_turnaround_time: float = 0.0
    avg_response_time: float = 0.0
    min_waiting_time: float = 0.0
    max_waiting_time: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0
    context_switches: int = 0
    avg_context_switches: float = 0.0
    total_idle_time: float = 0.0
    fairness_index: float = 0.0
    starvation_count: int = 0
    
    # Advanced metrics
    response_variance: float = 0.0
    tail_latency_p95: float = 0.0
    tail_latency_p99: float = 0.0
    scheduling_overhead: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'Algorithm': self.algorithm.value,
            'Avg Waiting Time': round(self.avg_waiting_time, 2),
            'Avg Turnaround Time': round(self.avg_turnaround_time, 2),
            'Avg Response Time': round(self.avg_response_time, 2),
            'CPU Utilization (%)': round(self.cpu_utilization * 100, 2),
            'Throughput': round(self.throughput, 2),
            'Avg Context Switches': round(self.avg_context_switches, 2),
            'Fairness Index': round(self.fairness_index, 2),
            'P95 Latency': round(self.tail_latency_p95, 2),
            'P99 Latency': round(self.tail_latency_p99, 2),
        }


@dataclass
class SchedulingResult:
    """
    Complete result of a scheduling simulation
    """
    algorithm: SchedulingAlgorithm
    processes: List[Process]
    metrics: AlgorithmMetrics
    timeline: List[Dict] = field(default_factory=list)
    execution_order: List[int] = field(default_factory=list)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'Total Processes': len(self.processes),
            'Total Time': max(p.completion_time for p in self.processes) if self.processes else 0,
            'Algorithm': self.algorithm.value,
            **self.metrics.to_dict()
        }
