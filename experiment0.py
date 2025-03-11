from Experiment import ExperimentMD
import torch

euclidean_test = ExperimentMD(lambda X: 10*5 + (X**2 -torch.cos(2*torch.pi*X)))
print(euclidean_test.run_experiment_minimise(5.0, 100, 0.001))
euclidean_test.create_minimisation_iteration_graph()