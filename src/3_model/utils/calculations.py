"""Class used to perform basic calculations and analysis on solution files"""

import os
import pickle

from data import ModelData


class Calculations:

    def __init__(self):
        # Model data
        self.data = ModelData()

    def discount_factor(self, year, base_year):
        """Discount factor"""

        # Discount factor
        discount_factor = 1 / ((1 + self.data.WACC) ** (year - base_year))

        return discount_factor

    def scenario_duration_days(self, year, scenario):
        """Get duration of operating scenarios (days)"""

        # Account for leap-years
        if year % 4 == 0:
            days_in_year = 366
        else:
            days_in_year = 365

        # Total days represented by scenario
        days = float(self.data.input_traces_dict[('K_MEANS', 'METRIC', 'NORMALISED_DURATION')][(year, scenario)]
                     * days_in_year)

        return days

    def get_total_discounted_investment_cost(self, m, iteration, investment_solution_dir):
        """Total discounted investment cost"""

        with open(os.path.join(investment_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            result = pickle.load(f)

        # Total discounted investment cost
        investment_cost = sum((self.discount_factor(y, m.Y.first()) / m.WACC) * m.GAMMA[g] * m.I_C[g, y] * result['x_c'][(g, y)]
                              for g in m.G_C for y in m.Y)

        return investment_cost

    def get_total_discounted_fom_cost(self, m, iteration, investment_solution_dir):
        """Total discounted FOM cost"""

        with open(os.path.join(investment_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            result = pickle.load(f)

        # FOM costs for candidate units
        candidate_fom = sum(self.discount_factor(y, m.Y.first()) * m.C_FOM[g] * result['a'][(g, y)] for g in m.G_C for y in m.Y)

        # FOM costs for existing units - note no FOM cost paid if unit retires
        existing_fom = sum(self.discount_factor(y, m.Y.first()) * m.C_FOM[g] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E for y in m.Y)

        # Expression for total FOM cost
        total_fom = candidate_fom + existing_fom

        return total_fom

    def get_end_of_horizon_fom_cost(self, m, iteration, investment_solution_dir):
        """Total discounted FOM cost after model horizon (assume cost in last year is paid in perpetuity)"""

        with open(os.path.join(investment_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            result = pickle.load(f)

        # FOM costs for candidate units
        candidate_fom = sum((self.discount_factor(m.Y.last(), m.Y.first()) / m.WACC) * m.C_FOM[g] * result['a'][(g, m.Y.last())]
                            for g in m.G_C)

        # FOM costs for existing units - note no FOM cost paid if unit retires
        existing_fom = sum((self.discount_factor(m.Y.last(), m.Y.first()) / m.WACC) * m.C_FOM[g] * m.P_MAX[g] * (1 - m.F[g, m.Y.last()])
                           for g in m.G_E)

        # Expression for total FOM cost
        total_fom = candidate_fom + existing_fom

        return total_fom

    @staticmethod
    def get_operating_scenario_cost(iteration, year, scenario, uc_solution_dir):
        """Get total operating cost"""

        with open(os.path.join(uc_solution_dir, f'uc-results_{iteration}_{year}_{scenario}.pickle'), 'rb') as f:
            result = pickle.load(f)

        # Objective value (not discounted, and not scaled by day weighting)
        objective = result['OBJECTIVE']

        return objective

    def get_total_discounted_operating_scenario_cost(self, iteration, uc_solution_dir):
        """Get total discounted cost for all operating scenarios"""

        # Files containing operating scenario results for a given iteration
        files = [f for f in os.listdir(uc_solution_dir) if f'uc-results_{iteration}' in f]

        # Initialise operating scenario total cost
        total_cost = 0

        for f in files:
            # Get year and scenario ID from filename
            year, scenario = int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

            # Get scenario duration
            duration = self.scenario_duration_days(year, scenario)

            # Discount factor
            discount = self.discount_factor(year, base_year=2016)

            # Nominal operating cost for given scenario
            nominal_cost = self.get_operating_scenario_cost(iteration, year, scenario, uc_solution_dir)

            # Discounted cost and scaled by number of days assigned to scenario
            total_cost += duration * discount * nominal_cost

        return total_cost

    def get_end_of_horizon_operating_cost(self, m, iteration, uc_solution_dir):
        """Get term representing operating cost to be paid in perpetuity after end of model horizon"""

        # Files containing scenario results for final year in model horizon
        files = [f for f in os.listdir(uc_solution_dir) if f'uc-results_{iteration}_{m.Y.last()}' in f]

        # Initialise total cost for final year
        total_cost = 0

        for f in files:
            # Get year and scenario ID from filename
            year, scenario = int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

            assert year == m.Y.last(), f'Year is not final year in model horizon: {year}'

            # Get scenario duration
            duration = self.scenario_duration_days(year, scenario)

            # Discount factor
            discount = self.discount_factor(year, base_year=m.Y.first())

            # Nominal operating cost for given scenario
            nominal_cost = self.get_operating_scenario_cost(iteration, year, scenario, uc_solution_dir)

            # Discounted cost and scaled by number of days assigned to scenario. Cost paid in perpetuity.
            total_cost += duration * (discount / m.WACC) * nominal_cost

        return total_cost

    def get_upper_bound(self, m, iteration, investment_solution_dir, uc_solution_dir):
        """Get constant cost term to include in Benders cut"""

        # Total investment cost
        inv_cost = self.get_total_discounted_investment_cost(m, iteration, investment_solution_dir)

        # Total FOM cost
        fom_cost = self.get_total_discounted_fom_cost(m, iteration, investment_solution_dir)

        # FOM cost beyond end of model horizon
        fom_cost_end = self.get_end_of_horizon_fom_cost(m, iteration, investment_solution_dir)

        # Operating cost over model horizon
        op_cost = self.get_total_discounted_operating_scenario_cost(iteration, uc_solution_dir)

        # Operating cost beyond end of model horizon
        op_cost_end = self.get_end_of_horizon_operating_cost(m, iteration, uc_solution_dir)

        # Total constant cost in Benders cut
        total_cost = inv_cost + fom_cost + fom_cost_end + op_cost + op_cost_end

        return total_cost
