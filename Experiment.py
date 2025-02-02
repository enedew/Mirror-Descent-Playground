import torch
import numpy as np 
from MirrorDescent import MirrorDescent
import plotly.graph_objects as plotly
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
torch.manual_seed(0)

class ExperimentMD():
    # results variable determines wheter the calculate_metrics function should calculate classification or regression metrics 
    # and whether this should be recorded/calculated or not 
    def __init__(self, objective = lambda x: x**2, bregman="EUCLID", results = 'R',
                  gradient_calculation = "Classical"):
        self.results = results 
        self.objective = objective 
        self.bregman = bregman 
        self.gradient_calculation = gradient_calculation
        self.criterion = torch.nn.MSELoss()
        self.losses = {
            "MSE" : torch.nn.MSELoss(),
            "MAE" : torch.nn.L1Loss(),
            "Huber" : torch.nn.HuberLoss()
        }
        self.dgfs = {
            'EUCLID' : lambda x: 0.5 * torch.sum(x**2),
            # domain x > 0 (probabilities) s.t. sum(x) = 1 (ideally)
            'KL' : lambda x: torch.sum((x+1e-8) * torch.log(x +1e-8)),
        }
        self.metrics = []
        self.loss_logs = [] 
        self.gradient_logs = []
        self.divergence_logs = []
        self.minimisation_guesses = []
        self.prediction_data = {}


    def build_simple_MLP(self, layers, neurons=10):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1,neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons,1)
        )

    def construct_optimiser(self, params, lr):
        self.optimiser = MirrorDescent(params, lr, self.bregman)


    def generate_data(self, lbound, ubound, n_samples):
        X = np.linspace(lbound, ubound, n_samples).astype(np.float32) 
        Y = self.objective(torch.tensor(X)) 
        X = torch.tensor(X).unsqueeze(1)
        Y = torch.tensor(Y).unsqueeze(1) 
        return X, Y 


    def train_model(self, X, Y, epochs=2000):
        for epoch in range(epochs):
            pred = self.model(X)
            loss = self.criterion(pred, Y)

            self.optimiser.zero_grad()
            loss.backward()
            
            # record gradient norms 
            self.calculate_record_gradient_norms()

            
            # save original parameters before stepping for bregman calculation
            old_params = []
            for group in self.optimiser.param_groups:
                for param in group['params']:
                    old_params.append(param.data.clone())

            # kept getting NaN params and predictions, implementing this stopped it but still getting
            # the error for KL.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimiser.step()

            for param in self.model.parameters():
                if torch.isnan(param).any():
                    print("WARNING: NaN detected in parameter")

            
            self.calculate_record_bregman_divergence(old_params)
            self.loss_logs.append(loss.item())

            # calculate metrics 
            if epoch % (epochs / 10) == 0:
                if not torch.isnan(pred).any():
                    metrics = self.calculate_metrics(Y, pred)
                    self.metrics.append([epoch] + metrics)
                    print(f"Epoch: {epoch}, Loss: {loss.item()}")

                else:
                    print("WARNING: NaN prediction values present")
    

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

        self.prediction_data = {
            'X': X.detach().cpu().numpy().flatten(),
            'Y_true': Y.detach().cpu().numpy().flatten(),
            'Y_pred': preds_final.detach().cpu().numpy().flatten()
        }

    def run_experiment_minimise(self, init, iter, lr):
        # minimises an objective function using the MirrorDescent optimiser
        # first convert the starting point to a tensor
        self.minimisation_guesses.append(init)
        x = torch.tensor([float(init)], requires_grad=True)
        self.construct_optimiser([x], lr)

        print("Optimiser constructed, now starting minimisation")
        for i in range(iter):
            self.optimiser.zero_grad()
            y = self.objective(x)
            y.backward()

            # recording gradient norms
            self.calculate_record_gradient_norms()

            # record current params before stepping for bregman divergence calculatation
            old_params = []
            for group in self.optimiser.param_groups:
                for param in group['params']:
                    old_params.append(param.data.clone())
            
            self.optimiser.step()

            # calculate and record the bregman divergence between params of this iteration and the next
            self.calculate_record_bregman_divergence(old_params)

            self.minimisation_guesses.append(x.item())
        print("Iterations for minimisation complete")
    
    def calculate_record_gradient_norms(self):
        grad_norm = 0.0
        # calculating the euclidean norm of the gradients
        for group in self.optimiser.param_groups:
            for param in group['params']:
                if param.grad is not None: 
                    grad_norm += param.grad.data.norm().item()**2
        grad_norm = grad_norm**0.5
        self.gradient_logs.append(grad_norm)

    def calculate_record_bregman_divergence(self, old_params):
        total_divergence = 0.0
        params = 0 
        idx = 0 
        psi = self.dgfs[self.bregman]
        # loop through each parameter recording the divergence across a step for each
        for group in self.optimiser.param_groups:
            for param in group['params']:
                # psi(x) - psi(y) - <psi'(y) , x - y>
                previous = old_params[idx]
                psi_y = psi(previous)                   
                psi_x = psi(param)

                # inner product
                product = torch.sum(self.optimiser.grad_psi(previous) * (psi_x - psi_y))


                # calculating the bregman divergence
                divergence = psi_x - psi_y - product 
                total_divergence += divergence

                params += 1
                idx +=1 
        avg_divergence = total_divergence / params 
        avg_divergence = avg_divergence.cpu().detach().numpy()
        self.divergence_logs.append(avg_divergence)


    def run_experiment_mlp(self, data_lbound, data_ubound, n_samples, layers, neurons, epochs, lr):
        # function runs an experiment to approximate a given input function using a simple multi layer perceptron with MirrorDescent as the optimiser
        # generate the data
        # this should vary on the bregman, for now just built for euclidean 
        X, Y = self.generate_data(data_lbound, data_ubound, n_samples)
        print("data generated")
        # construct the model, basic MLP right now, adding different options later down the line
        self.build_simple_MLP(layers, neurons)
        print("model constructed")
        # construct optimiser using MirrorDescent class
        self.construct_optimiser(self.model.parameters(), lr)
        print("optimiser constructed")
        # train the model 
        self.train_model(X, Y, epochs)
        print("training complete")
        # final predict 
        self.predict(X, Y)


    def create_optimisation_path_graph(self):
        fig = plotly.Figure()
        print(self.minimisation_guesses[:10])
        y_values_guess = [self.objective(torch.tensor(x)) for x in self.minimisation_guesses]
        # generating a line for the objective function
       
        x_line = np.linspace(min(self.minimisation_guesses)-5, max(self.minimisation_guesses)+5, 500)
        y_line = [self.objective(torch.tensor(x)) for x in x_line]
        fig.add_trace(plotly.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Objective Function'
        ))


        # plotting the guesses
        fig.add_trace(plotly.Scatter(
            x=self.minimisation_guesses,
            y=y_values_guess,
            mode="markers+lines",
            marker=dict(size=2, color="red", symbol="circle"),
            line=dict(color='red', width=2),
            name="optimiser guesses"
        ))

        fig.update_layout(
            title='Optimisation Path',
            xaxis_title='x',
            yaxis_title='f(x)',
            legend=dict(
            orientation="h",   
            yanchor="bottom",  
            y=1.02,             
            xanchor="right",   
            x=1,
            bgcolor="rgba(255,255,255,0.5)"  
        ))

        return fig


    def create_loss_curve(self):
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x=list(range(len(self.loss_logs))),
            y=self.loss_logs,
            mode="lines",
            name="Loss"
        ))
        fig.update_layout(
            title="Loss curve",
            xaxis_title="Epochs",
            yaxis_title="MSE Loss"
        )
        fig.update_yaxes(
            type="log"
        )
        return fig
    
    def create_gradient_norm_graph(self, mini=True):
        if mini==True:
            xaxis = "Iteration"
        else: 
            xaxis = "Epoch"
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x=list(range(len(self.gradient_logs))),
            y = self.gradient_logs,
            mode="lines",
            name="gradient norm"
        ))
        fig.update_layout(
            title = f"Gradient Norm over {xaxis}s",
            xaxis_title = xaxis,
            yaxis_title = "Gradient Norm"
        )
        return fig

    def create_divergence_graph(self, mini=True):
        if mini==True:
            xaxis = "Iteration"
        else: 
            xaxis = "Epoch"
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x = list(range(len(self.divergence_logs))),
            y = self.divergence_logs,
            mode = "lines",
            name = "bregman divergence"
        ))
        fig.update_layout(
            title="Bregman Divergence between current and next parameters",
            xaxis_title=f"{xaxis}s",
            yaxis_title="Divergence Value"

        )
        return fig

    def create_function_approximation_plot(self):
        fig = plotly.Figure()
        data = self.prediction_data
        fig.add_trace(plotly.Scatter(
            x=data['X'],
            y=data['Y_true'],
            mode='lines',
            name='True Function'
        ))
        fig.add_trace(plotly.Scatter(
            x=data['X'],
            y=data['Y_pred'],
            mode='markers+lines',
            name='MLP Approximation',
            marker=dict(size=2, opacity=0.7)
        ))
        fig.update_layout(
            title='Function Approximation',
            xaxis_title='Input',
            yaxis_title='Output',
            legend=dict(
            orientation="h",   
            yanchor="bottom",  
            y=1.02,             
            xanchor="right",   
            x=1,
            bgcolor="rgba(255,255,255,0.5)"  
        ))
        return fig





    
    
    
            



