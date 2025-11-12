"""
Streamlit UI Components
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.dataset import generate_synthetic_dataset, load_csv_dataset, create_sample_csv
from core.algorithms import FCFSScheduler, SJFScheduler, RoundRobinScheduler, PriorityScheduler
from core.metrics import create_summary_table, get_metric_statistics
from core.ml_model import SchedulerMLModel

def render_header():
    """Render application header"""
    st.set_page_config(page_title="CPU Scheduler Optimizer", layout="wide")
    st.title("ML-Based CPU Scheduling Optimizer")
    st.markdown("Using Random Forest to predict optimal scheduling algorithms")

def render_sidebar():
    """Render sidebar with input mode selection"""
    with st.sidebar:
        st.header("Configuration")
        input_mode = st.radio(
            "Select Input Mode",
            ["Single Process", "Batch CSV Upload", "Generate Synthetic Data"]
        )
        return input_mode

def get_single_process_input() -> pd.DataFrame:
    """Get single process input from user"""
    st.subheader("Enter Process Details")
    
    col1, col2 = st.columns(2)
    with col1:
        arrival = st.number_input("Arrival Time", min_value=0, value=0)
        burst = st.number_input("Burst Time", min_value=1, value=10)
    with col2:
        priority = st.slider("Priority (1=highest)", min_value=1, max_value=5, value=3)
        prev_burst = st.number_input("Previous Burst", min_value=0, value=0)
    
    time_quantum = st.number_input("Time Quantum (for RR)", min_value=1, value=5)
    
    data = {
        'Arrival_Time': [arrival],
        'Burst_Time': [burst],
        'Priority': [priority],
        'Prev_Burst': [prev_burst],
        'Time_Quantum': [time_quantum]
    }
    
    return pd.DataFrame(data)

def get_batch_input() -> pd.DataFrame:
    """Get batch input from CSV file"""
    st.subheader("Upload Process Data")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    use_synthetic = st.checkbox("Generate synthetic data if no file provided", value=True)
    n_synthetic = st.number_input("Number of synthetic processes", min_value=10, max_value=5000, value=1000)
    
    if uploaded_file is not None:
        df, errors = load_csv_dataset(uploaded_file)
        if errors:
            st.error(f"Errors loading file: {', '.join(errors)}")
            return None
        st.success(f"Loaded {len(df)} processes from file")
        return df
    elif use_synthetic:
        st.info(f"Generating {n_synthetic} synthetic processes...")
        return generate_synthetic_dataset(n_synthetic)
    else:
        st.warning("Please upload a CSV file or enable synthetic data generation")
        return None

def run_scheduling_algorithms(processes: pd.DataFrame) -> list:
    """Run all scheduling algorithms on the dataset"""
    time_quantum = int(processes['Time_Quantum'].mean()) if 'Time_Quantum' in processes.columns else 5
    
    results = [
        FCFSScheduler.schedule(processes),
        SJFScheduler.schedule(processes),
        RoundRobinScheduler.schedule(processes, time_quantum=time_quantum),
        PriorityScheduler.schedule(processes)
    ]
    
    return results

def display_results(results: list):
    """Display scheduling results"""
    st.subheader("Algorithm Performance Summary")
    
    summary_df = create_summary_table(results)
    st.dataframe(summary_df, use_container_width=True)
    
    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Summary CSV",
        data=csv,
        file_name="scheduling_summary.csv",
        mime="text/csv"
    )
    
    # Detailed metrics stats
    stats = get_metric_statistics(results)
    st.subheader("Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best for Waiting Time", stats['avg_waiting_time']['best_algorithm'])
    with col2:
        st.metric("Best for Throughput", stats['throughput']['best_algorithm'])
    with col3:
        st.metric("Best CPU Utilization", stats['cpu_utilization']['best_algorithm'])

def visualize_results(results: list):
    """Create visualizations of scheduling results"""
    st.subheader("Visualizations")
    
    # Prepare data
    algorithm_names = [r['name'] for r in results]
    avg_waiting = [r['metrics']['avg_waiting_time'] for r in results]
    avg_turnaround = [r['metrics']['avg_turnaround_time'] for r in results]
    cpu_util = [r['metrics']['cpu_utilization'] for r in results]
    throughput = [r['metrics']['throughput'] for r in results]
    
    # Scatter plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(avg_waiting, avg_turnaround, s=200, alpha=0.6)
    for i, name in enumerate(algorithm_names):
        ax1.annotate(name, (avg_waiting[i], avg_turnaround[i]), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Average Waiting Time')
    ax1.set_ylabel('Average Turnaround Time')
    ax1.set_title('Waiting Time vs Turnaround Time')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # CPU Utilization bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(algorithm_names, cpu_util, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_title('CPU Utilization per Algorithm')
    ax2.set_ylim([0, 105])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    st.pyplot(fig2)
    
    # Metric selection
    st.subheader("Compare Metrics Across Algorithms")
    metric_choice = st.selectbox(
        "Select metric to visualize",
        ["Average Waiting Time", "Average Turnaround Time", "Average Response Time", "Throughput"]
    )
    
    metric_map = {
        'Average Waiting Time': avg_waiting,
        'Average Turnaround Time': avg_turnaround,
        'Average Response Time': [r['metrics']['avg_response_time'] for r in results],
        'Throughput': throughput
    }
    
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(algorithm_names, metric_map[metric_choice], marker='o', linewidth=2, markersize=8)
    ax3.set_ylabel(metric_choice)
    ax3.set_title(f'{metric_choice} Across Algorithms')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

def train_and_display_ml_model(results_list: list):
    """Train ML model and display results"""
    st.subheader("Machine Learning Model - Algorithm Prediction")
    
    model = SchedulerMLModel()
    
    # For demo, we'll train on multiple variations of the same data
    train_data_list = []
    for _ in range(5):
        train_data_list.append(results_list)
    
    X, y = model.prepare_training_data(train_data_list)
    training_report = model.train(X, y)
    
    # Display accuracy
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{training_report['accuracy']:.2%}")
    
    # Feature importance
    importance_df = model.get_feature_importance()
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(importance_df['Feature'], importance_df['Importance'])
    ax_imp.set_xlabel('Importance')
    ax_imp.set_title('Feature Importance in ML Model')
    st.pyplot(fig_imp)
    
    # Classification report
    st.subheader("Classification Report")
    report_data = []
    for algo in training_report['class_names']:
        if algo in training_report['classification_report']:
            report_data.append({
                'Algorithm': algo,
                'Precision': f"{training_report['classification_report'][algo]['precision']:.2f}",
                'Recall': f"{training_report['classification_report'][algo]['recall']:.2f}",
                'F1-Score': f"{training_report['classification_report'][algo]['f1-score']:.2f}"
            })
    
    st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    conf_matrix = training_report['confusion_matrix']
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(conf_matrix, cmap='Blues')
    
    matrix_size = len(training_report['class_names'])
    ax_cm.set_xticks(range(matrix_size))
    ax_cm.set_yticks(range(matrix_size))
    ax_cm.set_xticklabels(training_report['class_names'], rotation=45, ha='right')
    ax_cm.set_yticklabels(training_report['class_names'])
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    
    for i in range(matrix_size):
        for j in range(matrix_size):
            text = ax_cm.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black", fontweight="bold")
    
    plt.colorbar(im, ax=ax_cm)
    st.pyplot(fig_cm)
