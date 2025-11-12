"""
Main Streamlit Application
"""
import streamlit as st
from ui.components import (
    render_header, render_sidebar, get_single_process_input,
    get_batch_input, run_scheduling_algorithms, display_results,
    visualize_results, train_and_display_ml_model
)

def main():
    """Main application flow"""
    render_header()
    
    input_mode = render_sidebar()
    
    # Get input data
    if input_mode == "Single Process":
        processes = get_single_process_input()
        if st.button("Schedule Single Process"):
            with st.spinner("Processing..."):
                results = run_scheduling_algorithms(processes)
                display_results(results)
                visualize_results(results)
    
    elif input_mode == "Batch CSV Upload":
        processes = get_batch_input()
        if processes is not None and st.button("Run Scheduling Analysis"):
            with st.spinner("Running all algorithms..."):
                results = run_scheduling_algorithms(processes)
                display_results(results)
                visualize_results(results)
                train_and_display_ml_model(results)
    
    elif input_mode == "Generate Synthetic Data":
        n_processes = st.slider("Number of processes", min_value=10, max_value=2000, value=500)
        if st.button("Generate and Analyze"):
            with st.spinner("Generating synthetic data and running analysis..."):
                from core.dataset import generate_synthetic_dataset
                processes = generate_synthetic_dataset(n_processes)
                results = run_scheduling_algorithms(processes)
                display_results(results)
                visualize_results(results)
                train_and_display_ml_model(results)

if __name__ == "__main__":
    main()
