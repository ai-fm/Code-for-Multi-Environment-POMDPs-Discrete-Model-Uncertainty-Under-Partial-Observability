Run_experiments.py contains the code to run the AB-HSVI algorithm on the models ordered per table in the paper. AB_HSVI.py contains the code to run the AB-HSVI algorithm. AB_HSVI_RockSample_experts.py contains the code to run the AB-HSVI algorithm on a ME-POMDP instance of the RockSample problem and on each POMDP in this ME-POMDP, and to compute the value each POMDP agent policy achieves on the other environments (other POMDPs in the ME-POMDP).
AB_HSVI_triviality_test.py contains the code to test whether a problem instance is trivial, meaning AB_HSVI can solve it within 30 seconds.

The Generate files contain the code to generate the model files that are used to run the experiments on. Parser.py is used to parse the generated models into the datastructures used in the AB-HSVI implementation.

The Models folder contains all model files used for the experiments. The Results folder contains all results for the experiments.

Note that running the AB-HSVI implementation requires Gurobi.