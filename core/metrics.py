"""
Metrics calculation and summary functions
"""
import pandas as pd
from typing import List, Dict

def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create a summary comparison table of all algorithms.
    
    Args:
        results: List of algorithm results
        
    Returns:
        DataFrame with algorithm comparison
    """
    summary_data = []
    
    for result in results:
        metrics = result['metrics']
        summary_data.append({
            'Algorithm': result['name'],
            'Avg Waiting Time': f"{metrics['avg_waiting_time']:.2f}",
            'Avg Turnaround Time': f"{metrics['avg_turnaround_time']:.2f}",
            'Avg Response Time': f"{metrics['avg_response_time']:.2f}",
            'Max Waiting Time': f"{metrics['max_waiting_time']:.2f}",
            'Throughput': f"{metrics['throughput']:.4f}",
            'CPU Utilization (%)': f"{metrics['cpu_utilization']:.2f}"
        })
    
    return pd.DataFrame(summary_data)

def extract_features_for_ml(results: List[Dict]) -> pd.DataFrame:
    """
    Extract features from algorithm results for ML training.
    
    Args:
        results: List of algorithm results
        
    Returns:
        DataFrame with features and target labels
    """
    data = []
    
    for result in results:
        metrics = result['metrics']
        data.append({
            'avg_waiting_time': metrics['avg_waiting_time'],
            'avg_turnaround_time': metrics['avg_turnaround_time'],
            'avg_response_time': metrics['avg_response_time'],
            'max_waiting_time': metrics['max_waiting_time'],
            'throughput': metrics['throughput'],
            'cpu_utilization': metrics['cpu_utilization'],
            'algorithm': result['name']
        })
    
    return pd.DataFrame(data)

def get_metric_statistics(results: List[Dict]) -> Dict:
    """Get detailed statistics for each metric across algorithms."""
    stats = {}
    
    metric_keys = ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'throughput', 'cpu_utilization']
    
    for metric_key in metric_keys:
        values = [result['metrics'][metric_key] for result in results]
        stats[metric_key] = {
            'best_algorithm': results[values.index(max(values) if metric_key == 'throughput' or metric_key == 'cpu_utilization' else min(values))]['name'],
            'best_value': max(values) if metric_key == 'throughput' or metric_key == 'cpu_utilization' else min(values),
            'worst_value': min(values) if metric_key == 'throughput' or metric_key == 'cpu_utilization' else max(values)
        }
    
    return stats
