"""Batch script to process input traces and construct scenarios"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import utils.process_input_traces
import utils.combine_traces
import utils.scenario_reduction


if __name__ == '__main__':
    # Directory containing output files (contains inputs from previous steps)
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Root data directory
    root_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Process input traces
    input_traces = utils.process_input_traces.main(root_data_directory, output_directory)

    # Combine all input traces
    dataset = utils.combine_traces.main(root_data_directory, output_directory)

    # Construct scenarios using k-means clustering algorithm
    df_scenario_centroids = utils.scenario_reduction.main(output_directory)
