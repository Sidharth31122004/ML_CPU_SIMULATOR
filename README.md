# ML-Based CPU Scheduling Optimizer

A comprehensive Python application that simulates CPU scheduling algorithms, evaluates their performance, and uses machine learning to predict the optimal algorithm for different workloads.

## Features

- **Four Scheduling Algorithms**
  - First Come First Served (FCFS)
  - Shortest Job First (SJF)
  - Round Robin (RR)
  - Priority Scheduling

- **Performance Metrics**
  - Average Waiting Time
  - Average Turnaround Time
  - Average Response Time
  - Throughput
  - CPU Utilization

- **Input Modes**
  - Single process manual entry
  - Batch CSV file upload
  - Synthetic dataset generation

- **Machine Learning**
  - Random Forest classifier to predict best algorithm
  - Feature importance analysis
  - Model accuracy and classification reports
  - Confusion matrix visualization

- **Visualizations**
  - Scatter plot: Waiting Time vs Turnaround Time
  - Bar chart: CPU Utilization comparison
  - Line graph: Configurable metric comparison

## Installation

1. Clone or download the project directory

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

### Running the Application

\`\`\`bash
streamlit run app.py
\`\`\`

The application will open in your default browser at `http://localhost:8501`

### Input Modes

#### Single Process Mode
1. Select "Single Process" from sidebar
2. Enter process parameters:
   - Arrival Time
   - Burst Time
   - Priority (1-5)
   - Previous Burst
   - Time Quantum
3. Click "Schedule Single Process"

#### Batch CSV Upload
1. Select "Batch CSV Upload" from sidebar
2. Upload a CSV file with columns: `Arrival_Time`, `Burst_Time`, `Priority`, `Prev_Burst`, `Time_Quantum`
3. Or enable "Generate synthetic data" to create test data
4. Click "Run Scheduling Analysis"

#### Synthetic Data Generation
1. Select "Generate Synthetic Data" from sidebar
2. Choose number of processes (10-2000)
3. Click "Generate and Analyze"

### CSV Format

The CSV file should contain the following columns:
- `Arrival_Time`: Time when process arrives in queue
- `Burst_Time`: CPU time required by the process
- `Priority`: Priority level (1=highest, 5=lowest)
- `Prev_Burst`: Previous burst time (for context)
- `Time_Quantum`: Time quantum for Round Robin algorithm

The application automatically normalizes column names, so variations like `arrival time`, `Arrival_Time`, or `arrival_time` will work.

### Example CSV

\`\`\`csv
Arrival_Time,Burst_Time,Priority,Prev_Burst,Time_Quantum
0,5,3,2,5
1,8,2,3,5
2,3,4,1,5
3,6,1,4,5
5,2,5,0,5
\`\`\`

## Architecture

### Core Modules

**core/dataset.py**
- `generate_synthetic_dataset()`: Creates random process data
- `load_csv_dataset()`: Loads and validates CSV files
- `create_sample_csv()`: Generates sample CSV files

**core/algorithms.py**
- `FCFSScheduler`: First Come First Served implementation
- `SJFScheduler`: Shortest Job First implementation
- `RoundRobinScheduler`: Round Robin implementation
- `PriorityScheduler`: Priority scheduling implementation
- `SchedulingMetrics`: Metric calculation utility

**core/metrics.py**
- `create_summary_table()`: Generates comparison table
- `extract_features_for_ml()`: Prepares data for ML
- `get_metric_statistics()`: Computes statistics

**core/ml_model.py**
- `SchedulerMLModel`: Random Forest classifier for algorithm prediction
- Model training with cross-validation
- Feature importance analysis

### UI Module

**ui/components.py**
- All Streamlit components and rendering functions
- Input handling and validation
- Result visualization

## Understanding the Output

### Performance Metrics Explained

- **Waiting Time**: Time process waits before execution starts
- **Turnaround Time**: Total time from arrival to completion
- **Response Time**: Time from arrival to first execution
- **Throughput**: Number of processes completed per unit time
- **CPU Utilization**: Percentage of time CPU is busy

### Machine Learning Model

The Random Forest model learns which algorithm performs best based on:
- Process arrival patterns
- Burst time distributions
- Priority distributions
- Overall system characteristics

The model's accuracy indicates how reliably it can predict the optimal algorithm for new workloads.

## Troubleshooting

### CSV Load Errors
- Ensure CSV has required columns (case-insensitive)
- Check that all numeric columns contain valid numbers
- Verify Priority values are between 1-5
- Ensure Burst Time values are positive

### Algorithm Scheduling Issues
- Round Robin requires positive time quantum
- Priority values should be 1-5
- Burst times should be positive integers
- Arrival times should be non-negative

## Requirements

- Python 3.9 or higher
- See requirements.txt for package versions

## Notes

- The ML model is trained on data from the current session
- For better model performance, run multiple different workloads
- CPU Utilization calculation assumes no I/O wait time
- Priority scheduling uses lower number = higher priority convention

## Future Enhancements

- Support for preemptive scheduling algorithms
- Dynamic time quantum optimization
- Multi-core CPU scheduling simulation
- Real-time performance monitoring
- Model persistence and retraining

## License

This project is provided as-is for educational purposes.
"# ML_CPU_SIMULATOR" 
