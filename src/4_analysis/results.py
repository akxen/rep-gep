"""Extract results from solution files"""

import os
import pickle

from analysis import AnalyseResults


class ResultsExtractor:
    def __init__(self):
        self.analysis = AnalyseResults()

    def extract_bau_results(self, results_dir, keys, output_dir=None):
        """Extract BAU model results"""

        # All REP files
        filenames = [f for f in os.listdir(results_dir) if 'bau_case' in f]

        # Container for results
        results = {}

        for f in filenames:

            # Extract information for each  key
            for k in keys:
                if k == 'YEAR_AVERAGE_PRICE':
                    r = self.analysis.get_average_prices(results_dir, f, None, 'PRICES', -1)
                    r = r.loc[:, 'average_price_real']

                else:
                    r = self.analysis.extract_results(results_dir, f, k, stage=None, iteration='max', model=None)

                # Append results to main container
                results[k] = r.to_dict()

        # Save results
        if output_dir is not None:
            filename = 'bau_results.pickle'
            self.save_results(results, filename, output_dir)

        return results

    def extract_rep_results(self, results_dir, keys, output_dir=None):
        """Extract REP model results"""

        # All REP files
        filenames = [f for f in os.listdir(results_dir) if 'rep' in f]

        # Container for results
        results = {}

        for f in filenames:
            print(f'Processing: {f}')
            # Get carbon price from filename
            carbon_price = int(f.split('-')[1].replace('.pickle', ''))

            if carbon_price not in results.keys():
                results[carbon_price] = {}

            # Extract information for each  key
            for k in keys:
                if k == 'YEAR_AVERAGE_PRICE':
                    r = self.analysis.get_average_prices(results_dir, f, 'stage_2_rep', 'PRICES', -1)
                    r = r.loc[:, 'average_price_real']

                else:
                    r = self.analysis.extract_results(results_dir, f, k, stage='stage_2_rep', iteration='max',
                                                      model=None)

                # Append results to main container
                results[carbon_price][k] = r.to_dict()

        # Save results
        if output_dir is not None:
            filename = 'rep_results.pickle'
            self.save_results(results, filename, output_dir)

        return results

    def extract_carbon_tax_results(self, results_dir, keys, output_dir=None):
        """Extract carbon tax results"""

        # All REP files
        filenames = [f for f in os.listdir(results_dir) if 'rep' in f]

        # Container for results
        results = {}

        for f in filenames:
            print(f'Processing: {f}')
            # Get carbon price from filename
            carbon_price = int(f.split('-')[1].replace('.pickle', ''))

            if carbon_price not in results.keys():
                results[carbon_price] = {}

            # Extract information for each  key
            for k in keys:
                if k == 'YEAR_AVERAGE_PRICE':
                    r = self.analysis.get_average_prices(results_dir, f, 'stage_1_carbon_tax', 'PRICES', -1)
                    r = r.loc[:, 'average_price_real']

                else:
                    r = self.analysis.extract_results(results_dir, f, k, stage='stage_1_carbon_tax', iteration='max',
                                                      model=None)

                # Append results to main container
                results[carbon_price][k] = r.to_dict()

        # Save results
        if output_dir is not None:
            filename = 'carbon_tax_results.pickle'
            self.save_results(results, filename, output_dir)

        return results

    def extract_price_targeting_results(self, filename_filter, results_dir, keys, output_dir=None):
        """Extract price targeting scenario model results"""

        # All REP files
        filenames = [f for f in os.listdir(results_dir) if filename_filter in f]

        # Container for results
        results = {}

        for f in filenames:
            print(f'Processing: {f}')
            # Get carbon price and transition year from filename
            transition_year = int(f.split('-')[1].replace('_cp', ''))
            carbon_price = int(f.split('-')[2].replace('.pickle', ''))

            if transition_year not in results.keys():
                results[transition_year] = {}

            if carbon_price not in results[transition_year].keys():
                results[transition_year][carbon_price] = {}

            # Extract information for each  key
            for k in keys:
                if k == 'YEAR_AVERAGE_PRICE':
                    r = self.analysis.get_average_prices(results_dir, f, 'stage_3_price_targeting', 'PRICES', -1)
                    r = r.loc[:, 'average_price_real']

                else:
                    r = self.analysis.extract_results(results_dir, f, k, stage='stage_3_price_targeting',
                                                      iteration='max', model='primal')

                # Append results to main container
                results[transition_year][carbon_price][k] = r.to_dict()

        # Save results
        if output_dir is not None:
            filename = f'{filename_filter}_results.pickle'
            self.save_results(results, filename, output_dir)

        return results

    @staticmethod
    def save_results(results, filename, output_dir):
        """Save model results"""

        print(f'Saving results: {filename}')
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

    def extract_all_results(self, results_dir, output_dir):
        """Extract results for different scenarios"""

        print('Extracting BAU results')
        bau_results_keys = ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE', 'x_c']
        self.extract_bau_results(results_dir, bau_results_keys, output_dir)

        # Extract these keys for all other scenarios
        result_keys = ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE', 'YEAR_CUMULATIVE_SCHEME_REVENUE', 'x_c',
                       'YEAR_SCHEME_EMISSIONS_INTENSITY', 'YEAR_EMISSIONS_INTENSITY']

        print('Extracting tax results')
        self.extract_carbon_tax_results(results_dir, result_keys, output_dir)

        print('Extracting REP results')
        self.extract_rep_results(results_dir, result_keys, output_dir)

        print('Extracting heuristic results')
        for i in ['heuristic_baudev', 'heuristic_ptar', 'heuristic_pdev']:
            self.extract_price_targeting_results(i, results_dir, result_keys, output_dir)

    @staticmethod
    def load_results(directory, filename):
        """Load results"""

        with open(os.path.join(directory, filename), 'rb') as f:
            results = pickle.load(f)

        return results

    def combine_results(self, output_dir):
        """Combine extracted results"""

        # Load BAU, tax, and REP results
        bau = self.load_results(output_dir, 'bau_results.pickle')
        tax = self.load_results(output_dir, 'carbon_tax_results.pickle')
        rep = self.load_results(output_dir, 'rep_results.pickle')

        # Load price targeting results
        price_deviation = self.load_results(output_dir, 'heuristic_pdev_results.pickle')
        bau_deviation = self.load_results(output_dir, 'heuristic_baudev_results.pickle')
        trajectory_deviation = self.load_results(output_dir, 'heuristic_ptar_results.pickle')

        # Combine into single dictionary
        combined = {'bau': bau, 'tax': tax, 'rep': rep, 'pdev': price_deviation, 'baudev': bau_deviation,
                    'ptar': trajectory_deviation}

        # Save combine results
        with open(os.path.join(output_dir, 'model_results.pickle'), 'wb') as f:
            pickle.dump(combined, f)

        return combined


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'tmp', 'local')

    # Object used to parse and extract model results
    extractor = ResultsExtractor()

    # Extract results from all scenarios and save in tmp directory
    # extractor.extract_all_results(results_directory, output_directory)
    
    # Combine model results into single dictionary
    model_results = extractor.combine_results(output_directory)
