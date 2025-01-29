import torch
import numpy as np 

class ExperimentMD():
    # results = vector of binary values, each one corresponding to a result or metric
    # and whether this should be recorded/calculated or not 
    def __init__(self, objective, bregman="EUCLID", results = [], gradient_calculation = "Classicical"):
        self.results = results 
        self.objective = objective 

