import pickle

with open('output/operational_plan/uc-results_1_2016_1.pickle', 'rb') as f:
    results = pickle.load(f)

print(results['CANDIDATE_CAPACITY_FIXED'])
iterations = [1, 2]

