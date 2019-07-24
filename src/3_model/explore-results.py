"""Check model results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import matplotlib.pyplot as plt

from utils.master import InvestmentPlan
from utils.calculations import Calculations


def load_investment_results(iteration):
    """Load investment results"""

    with open(f'output/investment_plan/investment-results_{iteration}.pickle', 'rb') as f:
        results = pickle.load(f)

    return results


def load_dispatch_results(iteration, year, scenario):
    """Load scenario results"""

    with open(f'output/dispatch_plan/uc-results_{iteration}_{year}_{scenario}.pickle', 'rb') as f:
        results = pickle.load(f)

    return results


def plot_results(series, title):
    """Plot results"""

    fig, ax = plt.subplots()
    ax.plot(series)
    ax.set_title(title)
    plt.show()


def plot_prices(iteration, year, scenario, zone='NCEN'):
    """Check price results from all UC models"""

    # Load results
    results = load_dispatch_results(iteration, year, scenario)

    # Data to plot
    series = results['PRICES'][zone].values()

    # Title for plot
    title = f'PRICES - {zone} - {iteration} - {year} - {scenario}'

    plot_results(series, title)


def plot_all_prices(subproblem_results_dir):
    """Plot all prices"""

    files = [f for f in os.listdir(subproblem_results_dir) if '.pickle' in f]

    for f in files:
        # Iteration ID, year, and scenario
        it, year, scen = int(f.split('_')[-3]), int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

        plot_prices(it, year, scen)


def plot_dispatch(iteration, year, scenario, generator='TARONG#2'):
    """Plot energy output"""

    # Load results
    results = load_dispatch_results(iteration, year, scenario)

    # Data to plot
    series = results['ENERGY'][generator].values()

    # Title for plot
    title = f'ENERGY - {generator} - {iteration} - {year} - {scenario}'

    plot_results(series, title)


def plot_all_dispatch(subproblem_results_dir):
    """Plot all prices"""

    files = [f for f in os.listdir(subproblem_results_dir) if '.pickle' in f]

    for f in files:
        # Iteration ID, year, and scenario
        it, year, scen = int(f.split('_')[-3]), int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

        plot_dispatch(it, year, scen)


def check_solve_status(iteration, year, scenario):
    """Check solution status"""

    # Load results
    results = load_dispatch_results(iteration, year, scenario)

    if results['SOLVE_STATUS']['Solver'][0]['Status'].key == 'ok':
        return True
    else:
        return False


def check_all_solve_statuses(subproblem_results_dir):
    """Check solve status for all dispatch subproblems"""

    files = [f for f in os.listdir(subproblem_results_dir) if '.pickle' in f]

    for f in files:
        # Iteration ID, year, and scenario
        it, year, scen = int(f.split('_')[-3]), int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

        if check_solve_status(it, year, scen):
            print(f'OK: {it}-{year}-{scen}')
        else:
            print(f'Bad solve status for: {it}-{year}-{scen}')


def compare_iterations():
    """Compare objective function value"""

    for year in [2016, 2017]:
        for scenario in range(1, 11):
            results_1 = load_dispatch_results(1, year, scenario)
            objective_1 = results_1['OBJECTIVE']

            results_2 = load_dispatch_results(2, year, scenario)
            objective_2 = results_2['OBJECTIVE']

            if objective_1 == objective_2:
                print(f'{year}-{scenario} Objectives match')
            else:
                print(f'{year}-{scenario} MISMATCH')


if __name__ == '__main__':
    # Object used to perform calculations
    calculations = Calculations()

    # Object used to construct Benders master problem
    inv = InvestmentPlan()

    # Master problem
    master = inv.construct_model()

    # Define iteration, year, and scenario
    i, y, s = 1, 2016, 1

    # Dispatch results
    r = load_dispatch_results(i, y, s)

    subproblem_results_dir = os.path.join(os.path.dirname(__file__), 'output', 'dispatch_plan')

    # Plot all price results
    # plot_all_prices(subproblem_results_dir)
    #
    # Plot all dispatch data
    # plot_all_dispatch(subproblem_results_dir)
    #
    # Check solve status
    check_all_solve_statuses(subproblem_results_dir)

    # compare_iterations()

    # upper = calculations.get_upper_bound(master, 2, 'output/investment_plan', 'output/dispatch_plan')
    # op_cost = calculations.get_total_discounted_operating_scenario_cost(2, 'output/dispatch_plan')
    # op_cost_eoh = calculations.get_end_of_horizon_operating_cost(master, 2, 'output/dispatch_plan')
