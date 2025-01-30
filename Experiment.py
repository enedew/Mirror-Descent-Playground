import torch
import numpy as np 
from MirrorDescent import MirrorDescent
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
torch.manual_seed(0)

class ExperimentMD():
    # results variable determines wheter the calculate_metrics function should calculate classification or regression metrics 
    # and whether this should be recorded/calculated or not 
    def __init__(self, objective = lambda x: x**2, bregman="EUCLID", results = 'R',
                  gradient_calculation = "Classical", criterion=torch.nn.MSELoss()):
        self.results = results 
        self.objective = objective 
        self.bregman = bregman 
        self.gradient_calculation = gradient_calculation
        self.criterion = criterion
        self.metrics = []
        self.loss_logs = [] 


    def build_simple_MLP(self, neurons=10):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1,neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons,1)
        )

    def construct_optimiser(self, params, lr):
        self.optimiser = MirrorDescent(params, lr, self.bregman)


    def generate_data(self, lbound, ubound, n_samples):
        X = np.linspace(lbound, ubound, n_samples).astype(np.float32) 
        Y = self.objective(X) 
        X = torch.tensor(X).unsqueeze(1)
        Y = torch.tensor(Y).unsqueeze(1) 
        return X, Y 


    def train_model(self, X, Y, epochs=2000):
        for epoch in range(epochs):
            pred = self.model(X)
            loss = self.criterion(pred, Y)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.loss_logs.append(loss.item())

            if epoch % (epochs / 10) == 0:
                metrics = self.calculate_metrics(Y, pred)
                self.metrics.append([epoch] + metrics)
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
    

    def calculate_metrics(self, true, pred):
    
        true = true.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        if self.results == 'R':
            metric_functions = [mean_absolute_error, r2_score]
        elif self.results == 'C':
            metric_functions = [accuracy_score, precision_score, recall_score, f1_score]
        metrics = [calculate_metric(true, pred) for calculate_metric in metric_functions]
        

        return metrics
        
    def predict(self, X, Y):
        preds_final = self.model(X)
        loss_final = self.criterion(preds_final, Y)
        final_metrics = self.calculate_metrics(preds_final, Y) 
        self.metrics.append(['T'] + final_metrics)
        self.loss_logs.append(loss_final.item())

    def run_experiment(self, epochs, lr):
        # generate the data
        # this should vary on the bregman, for now just built for euclidean 
        X, Y = self.generate_data(-5, 5, 500)
        print("data generated")
        # construct the model, basic MLP right now, adding different options later down the line
        self.build_simple_MLP()
        print("model constructed")
        # construct optimiser using MirrorDescent class
        self.construct_optimiser(self.model.parameters(), lr)
        print("optimiser constructed")
        # train the model 
        self.train_model(X, Y, epochs)
        print("training complete")
        # final predict 
        self.predict(X, Y)





    
    
    
            



