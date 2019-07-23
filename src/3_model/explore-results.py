import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import pickle

from utils.benders_master import InvestmentPlan

# with open('output/operational_plan/uc-results_1_2016_1.pickle', 'rb') as f:
#     results = pickle.load(f)

with open('output/investment_plan/investment-results_2.pickle', 'rb') as f:
    r1 = pickle.load(f)

with open('output/investment_plan/investment-results_3.pickle', 'rb') as f:
    r2 = pickle.load(f)


master = InvestmentPlan()
model_inv = master.construct_model()

investment_solution_dir = os.path.join(os.path.dirname(__file__), 'output', 'investment_plan')
uc_solution_dir = os.path.join(os.path.dirname(__file__), 'output', 'operational_plan')

master.get_cost_upper_bound(model_inv, 3, investment_solution_dir, uc_solution_dir)