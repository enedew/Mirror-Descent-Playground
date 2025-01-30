from Experiment import ExperimentMD

euclidean_test = ExperimentMD(lambda X: X**2 + 3*X + 5)
euclidean_test.run_experiment(5000, 0.01)