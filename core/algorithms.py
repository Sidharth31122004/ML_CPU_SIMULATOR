"""
CPU Scheduling Algorithms Implementation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class SchedulingMetrics:
    """Container for scheduling metrics"""
    def __init__(self):
        self.waiting_times = []
        self.turnaround_times = []
        self.response_times = []
        self.completion_times = []
        
    def compute_metrics(self) -> Dict:
        """Compute aggregate metrics"""
        return {
            'avg_waiting_time': np.mean(self.waiting_times) if self.waiting_times else 0,
            'avg_turnaround_time': np.mean(self.turnaround_times) if self.turnaround_times else 0,
            'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
            'max_waiting_time': np.max(self.waiting_times) if self.waiting_times else 0,
            'throughput': len(self.completion_times) / (max(self.completion_times) - min(self.completion_times)) if len(self.completion_times) > 1 else 0,
        }

class FCFSScheduler:
    """First Come First Served Scheduling"""
    
    @staticmethod
    def schedule(processes: pd.DataFrame) -> Dict:
        """Execute FCFS algorithm"""
        df = processes.sort_values('Arrival_Time').reset_index(drop=True)
        metrics = SchedulingMetrics()
        
        current_time = 0
        detail_records = []
        
        for idx, row in df.iterrows():
            arrival = row['Arrival_Time']
            burst = row['Burst_Time']
            
            # Process arrives after current time, idle CPU
            if arrival > current_time:
                current_time = arrival
            
            response_time = current_time - arrival
            completion_time = current_time + burst
            turnaround_time = completion_time - arrival
            waiting_time = turnaround_time - burst
            
            metrics.waiting_times.append(waiting_time)
            metrics.turnaround_times.append(turnaround_time)
            metrics.response_times.append(response_time)
            metrics.completion_times.append(completion_time)
            
            detail_records.append({
                'Process': idx,
                'Arrival': arrival,
                'Burst': burst,
                'Waiting': waiting_time,
                'Turnaround': turnaround_time,
                'Completion': completion_time
            })
            
            current_time = completion_time
        
        agg_metrics = metrics.compute_metrics()
        agg_metrics['cpu_utilization'] = (sum(df['Burst_Time']) / current_time * 100) if current_time > 0 else 0
        
        return {
            'name': 'FCFS',
            'metrics': agg_metrics,
            'details': pd.DataFrame(detail_records)
        }

class SJFScheduler:
    """Shortest Job First Scheduling (Non-preemptive)"""
    
    @staticmethod
    def schedule(processes: pd.DataFrame) -> Dict:
        """Execute SJF algorithm"""
        df = processes.sort_values('Arrival_Time').reset_index(drop=True)
        metrics = SchedulingMetrics()
        
        current_time = 0
        completed = set()
        detail_records = []
        
        while len(completed) < len(df):
            # Find available processes (arrived and not completed)
            available = df[(df['Arrival_Time'] <= current_time) & (~df.index.isin(completed))]
            
            if available.empty:
                # No process available, move to next arrival
                next_arrival = df[~df.index.isin(completed)]['Arrival_Time'].min()
                current_time = next_arrival
                available = df[(df['Arrival_Time'] <= current_time) & (~df.index.isin(completed))]
            
            # Select process with shortest burst time
            idx = available['Burst_Time'].idxmin()
            row = df.loc[idx]
            
            response_time = current_time - row['Arrival_Time']
            completion_time = current_time + row['Burst_Time']
            turnaround_time = completion_time - row['Arrival_Time']
            waiting_time = turnaround_time - row['Burst_Time']
            
            metrics.waiting_times.append(waiting_time)
            metrics.turnaround_times.append(turnaround_time)
            metrics.response_times.append(response_time)
            metrics.completion_times.append(completion_time)
            
            detail_records.append({
                'Process': idx,
                'Arrival': row['Arrival_Time'],
                'Burst': row['Burst_Time'],
                'Waiting': waiting_time,
                'Turnaround': turnaround_time,
                'Completion': completion_time
            })
            
            current_time = completion_time
            completed.add(idx)
        
        agg_metrics = metrics.compute_metrics()
        agg_metrics['cpu_utilization'] = (sum(df['Burst_Time']) / current_time * 100) if current_time > 0 else 0
        
        return {
            'name': 'SJF',
            'metrics': agg_metrics,
            'details': pd.DataFrame(detail_records)
        }

class RoundRobinScheduler:
    """Round Robin Scheduling"""
    
    @staticmethod
    def schedule(processes: pd.DataFrame, time_quantum: int = 5) -> Dict:
        """Execute Round Robin algorithm"""
        df = processes.sort_values('Arrival_Time').reset_index(drop=True)
        df = df.copy()
        df['Remaining_Burst'] = df['Burst_Time']
        
        metrics = SchedulingMetrics()
        current_time = 0
        queue = []
        detail_records = []
        response_times_recorded = set()
        
        idx_ptr = 0
        
        while True:
            # Add all arrived processes to queue
            while idx_ptr < len(df) and df.iloc[idx_ptr]['Arrival_Time'] <= current_time:
                queue.append(idx_ptr)
                idx_ptr += 1
            
            if not queue and idx_ptr < len(df):
                # No process in queue, jump to next arrival
                current_time = df.iloc[idx_ptr]['Arrival_Time']
                queue.append(idx_ptr)
                idx_ptr += 1
            elif not queue:
                break
            
            # Process front of queue
            process_idx = queue.pop(0)
            row = df.iloc[process_idx]
            
            # Record response time (only on first execution)
            if process_idx not in response_times_recorded:
                metrics.response_times.append(current_time - row['Arrival_Time'])
                response_times_recorded.add(process_idx)
            
            # Execute for time quantum or remaining time
            exec_time = min(time_quantum, df.iloc[process_idx]['Remaining_Burst'])
            current_time += exec_time
            df.iloc[process_idx, df.columns.get_loc('Remaining_Burst')] -= exec_time
            
            # If process not finished, add back to queue
            if df.iloc[process_idx]['Remaining_Burst'] > 0:
                queue.append(process_idx)
            else:
                # Process completed
                completion_time = current_time
                turnaround_time = completion_time - row['Arrival_Time']
                waiting_time = turnaround_time - row['Burst_Time']
                
                metrics.waiting_times.append(waiting_time)
                metrics.turnaround_times.append(turnaround_time)
                metrics.completion_times.append(completion_time)
                
                detail_records.append({
                    'Process': process_idx,
                    'Arrival': row['Arrival_Time'],
                    'Burst': row['Burst_Time'],
                    'Waiting': waiting_time,
                    'Turnaround': turnaround_time,
                    'Completion': completion_time
                })
        
        agg_metrics = metrics.compute_metrics()
        agg_metrics['cpu_utilization'] = (sum(df['Burst_Time']) / current_time * 100) if current_time > 0 else 0
        
        return {
            'name': 'Round Robin',
            'metrics': agg_metrics,
            'details': pd.DataFrame(detail_records)
        }

class PriorityScheduler:
    """Priority Scheduling (Non-preemptive, lower number = higher priority)"""
    
    @staticmethod
    def schedule(processes: pd.DataFrame) -> Dict:
        """Execute Priority Scheduling algorithm"""
        df = processes.sort_values('Arrival_Time').reset_index(drop=True)
        metrics = SchedulingMetrics()
        
        current_time = 0
        completed = set()
        detail_records = []
        
        while len(completed) < len(df):
            # Find available processes
            available = df[(df['Arrival_Time'] <= current_time) & (~df.index.isin(completed))]
            
            if available.empty:
                # No process available, move to next arrival
                next_arrival = df[~df.index.isin(completed)]['Arrival_Time'].min()
                current_time = next_arrival
                available = df[(df['Arrival_Time'] <= current_time) & (~df.index.isin(completed))]
            
            # Select process with highest priority (lowest priority number)
            idx = available['Priority'].idxmin()
            row = df.loc[idx]
            
            response_time = current_time - row['Arrival_Time']
            completion_time = current_time + row['Burst_Time']
            turnaround_time = completion_time - row['Arrival_Time']
            waiting_time = turnaround_time - row['Burst_Time']
            
            metrics.waiting_times.append(waiting_time)
            metrics.turnaround_times.append(turnaround_time)
            metrics.response_times.append(response_time)
            metrics.completion_times.append(completion_time)
            
            detail_records.append({
                'Process': idx,
                'Arrival': row['Arrival_Time'],
                'Priority': row['Priority'],
                'Burst': row['Burst_Time'],
                'Waiting': waiting_time,
                'Turnaround': turnaround_time,
                'Completion': completion_time
            })
            
            current_time = completion_time
            completed.add(idx)
        
        agg_metrics = metrics.compute_metrics()
        agg_metrics['cpu_utilization'] = (sum(df['Burst_Time']) / current_time * 100) if current_time > 0 else 0
        
        return {
            'name': 'Priority',
            'metrics': agg_metrics,
            'details': pd.DataFrame(detail_records)
        }
