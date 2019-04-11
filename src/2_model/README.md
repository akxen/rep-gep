# Model strucuture
Goal is to identify the optimal trajectory for the emissions intensity baseline and the permit price in the context of a generation expansion planning framework. Unit commitment decisions should also be captured. 

Representative time segments will be used to capture system operation for each year in the planning horizon, with a nested decomposition approach used to find the optimal trajectory of policy parameters. A rolling-window approach will be used to solve the unit commitment problem for each representative time period within each year of the planning horizon. Information from these subproblems will then be fed up to the main objective which includes generation investment decisions. This follows the basic Benders decomposition framework. The optimal investment plan will be determined for a given set of policy parameters. Can also treat the emissions price as a complicating variable. Find the path of the emissions price that leads to the least-cost generation mix that satisfies the cumulative and terminal emissions constraints. Then, employ a heuristic approach to identify to the path of the emissions intensity baseline that minimises deviations to the average electricity price. Can identify marginal generators in each period and modify their costs accordingly (they will still be marginal). Goal is to minimise price deviations between successive periods. Construct model so it can be run up until 2050. Demand traces, wind traces, solar traces. Canditate generating technologies - allow coal, coal with CCS, gas, wind, and storage. Each has 3 different sizes. Can be invested in each NEM zone. 

## Modules
#### Data collection (DataHandler.py class=ModelData)
Load and format data so it can be used in UC models.

#### UC model (SubProblem.py, class=UC)
Instantiate UC model. Contains methods that update model parameters.
1. Method to set model parameters for a given time period
2. Method to fix variables at start of window. Fixes first interval when starting rolling window (use initialised values). Fixes multiple periods (specified by window overlap) when running rolling window.
3. Method to fix binary variables and solve linear program to obtain dual variables (prices)
4. Method to output required variable values to be used in next iteration of rolling window approach.

#### Run UC model (SubProblem.py, class=RunUC)
Run UC model for a given period of time. Implement rolling window approach. Save relevant data in a standard format.
This should return dual variable information for each complicating variable. Use this to construct Benders cuts. Need to handle possibility of model infeasibility. Also keep track of marginal units in each subproblem. Must accept arguments that define policy paramters for the whole interval, and fixes variables correctly when implementing the rolling window solution protocol. Should also store model output.

#### Master problem (MasterProblem.py, class=Master)
Use information obtained from sub-problem solutions (obtained using RunUC.py) when formulating the master problem.

#### Run Master problem (MasterProblem.py, class=RunMaster.py)
Add benders cuts and run master problem.

#### Emissions intensity baseline problem (Policy.py, class=Baseline)
Find emissions intensity baseline trajectory that minimises price deviations subject to a revenue constraint. Can identify marginal units in each subproblem, and the level of demand for that period. Use this information to compute the level of the baseline that minimises price deviations.

#### Benders decomposition (Benders.py, class=RunBenders)
Implement the Benders decomposition algorithm to find the optimal investment plan and permit price path.

#### Check solution (Check.py, class=CheckSolution)
Run model using fixed baseline policy parameters. Check solution is consistent with what was found previously. Need to check that the baseline does not influence investment decisions. Could probably do this earlier. Need to confirm that policy parameters are decoupled when implementing program.

