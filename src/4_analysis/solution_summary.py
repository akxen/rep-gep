"""Parse logs to figure out solution time"""

import os
import json

import pandas as pd


class SolutionSummary:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_solution_summary(self):
        """Get solution summary information"""

        # Container for lines containing solution summary information
        lines = []

        # Parse file
        with open(os.path.join(self.output_dir, 'solution_summary.txt'), 'r') as f:
            for l in f.readlines():
                lines.append(json.loads(l.replace('\'', '\"').replace('\n','')))

        # Convert to DataFrame
        df = pd.DataFrame(lines)

        return df


if __name__ == '__main__':
    # Directory containing model output files
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')

    # Object used to extract solution summary information
    solution = SolutionSummary(output_directory)

    # Solution summary
    s = solution.get_solution_summary()


