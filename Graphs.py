import plotly.graph_objects as plotly 
import numpy as np
import torch 

class Graphs(): 
    # Class for constructing and updating graphs
    # all methods starting with "create" contain the initial constructing of a graph
    # all methods starting with "add" update an existing graph with a new experiments results 

    def __init__(self): 
        self.line_colors = {
            1 : "#1A8FE3",
            2 : "#6610F2",
            3 : "#D11149",
            4 : "#F17105", 
            5 : "#E6C229",
        }

    
    def create_loss_curve(self, loss_logs):
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x=list(range(len(loss_logs))),
            y=loss_logs,
            mode="lines",
            opacity=0.7,
            line=dict(color=self.line_colors[1]),
            name="1"
        ))
        fig.update_layout(
            title="Loss curve",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            )
        )
        fig.update_yaxes(
            type="log"
        )
        self.loss_curve_graph = fig 
        return self.loss_curve_graph 
    
   
    def add_loss_curve(self, loss_logs, exp_number): 
        if self.loss_curve_graph is not None: 
            self.loss_curve_graph.add_trace(plotly.Scatter(
            x=list(range(len(loss_logs))),
            y = loss_logs,
            mode="lines",
            opacity=0.7,
            line=dict(color=self.line_colors[exp_number]),
            name=f'{exp_number}'
            )) 
            return self.loss_curve_graph
        else: 
            raise ValueError("loss graph does not exist.")

    def create_gradient_norm_graph(self, gradient_logs):
        xaxis="Iteration"
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x=list(range(len(gradient_logs))),
            y = gradient_logs,
            mode="lines",
            name="1",
            opacity=0.7,
            line=dict(color=self.line_colors[1]),
        ))
        fig.update_layout(
            title = f"Gradient Norm over {xaxis}s",
            xaxis_title = xaxis,
            yaxis_title = "Gradient Norm",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            )
        )
        self.gradient_norm_graph = fig 
        return self.gradient_norm_graph
    
    def add_gradient_norm(self, gradient_logs, exp_number):
        if self.gradient_norm_graph is not None: 
            self.gradient_norm_graph.add_trace(plotly.Scatter(
            x=list(range(len(gradient_logs))),
            y = gradient_logs,
            mode="lines",
            opacity=0.7,
            line=dict(color=self.line_colors[exp_number]),
            name=f'{exp_number}'
            )) 
            return self.gradient_norm_graph
        else: 
            raise ValueError("gradient norm graph does not exist.")

    
    def create_divergence_graph(self, divergence_logs):
        xaxis="Iteration"
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x = list(range(len(divergence_logs))),
            y = divergence_logs,
            opacity=0.7,
            mode = "lines",
            name = "1",
            line=dict(color=self.line_colors[1])
        ))
        fig.update_layout(
            title="Bregman Divergence of parameters",
            xaxis_title=f"{xaxis}s",
            yaxis_title="Divergence Value",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            )
        )
        self.divergence_graph = fig 
        return self.divergence_graph 
    
    def add_divergence(self, divergence_logs, exp_number): 
        if self.divergence_graph is not None: 
            self.divergence_graph.add_trace(plotly.Scatter(
            x=list(range(len(divergence_logs))),
            y = divergence_logs,
            mode="lines",
            opacity=0.7,
            line=dict(color=self.line_colors[exp_number]),
            name=f'{exp_number}'
            )) 
            return self.divergence_graph
        else: 
            raise ValueError("divergence graph does not exist.")
        
    
    def create_function_approximation_plot(self, prediction_data):
        fig = plotly.Figure()
        fig.add_trace(plotly.Scatter(
            x=prediction_data['X'],
            y=prediction_data['Y_true'],
            mode='lines',
            name='True Function',
            line=dict(color="black")
        ))
        fig.add_trace(plotly.Scatter(
            x=prediction_data['X'],
            y=prediction_data['Y_pred'],
            mode='markers+lines',
            name='1',
            opacity=0.5,
            line=dict(color=self.line_colors[1]),
            marker=dict(size=2, opacity=0.5)
        ))
        fig.update_layout(
            title='Function Approximation',
            xaxis_title='Input',
            yaxis_title='Output',
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            ),
            legend=dict(
            orientation="h",   
            yanchor="bottom",  
            y=1.02,             
            xanchor="right",   
            x=1,
            bgcolor="rgba(255,255,255,0.5)"  
        ))
        self.approximation_graph = fig 
        return self.approximation_graph 
    
    def add_function_approximation(self, prediction_data, exp_number): 
        if self.approximation_graph is not None: 
            self.approximation_graph.add_trace(plotly.Scatter(
            x=prediction_data['X'],
            y=prediction_data['Y_pred'],
            mode='markers+lines',
            name=f'{exp_number}',
            opacity=0.5,
            line=dict(color=self.line_colors[exp_number]),
            marker=dict(size=2, opacity=0.5)
        ))
            return self.approximation_graph
        else: 
            raise ValueError("approximation graph does not exist.")
        
    def create_optimisation_path_graph(self, minimisation_guesses, objective):
        self.optimisation_path_graph = plotly.Figure()
        print(minimisation_guesses[:10])
        y_values_guess = [objective(torch.tensor(x)) for x in minimisation_guesses]
        # generating a line for the objective function
       
        x_line = np.linspace(min(minimisation_guesses)-5, max(minimisation_guesses)+5, 500)
        y_line = [objective(torch.tensor(x)) for x in x_line]
        self.optimisation_path_graph.add_trace(plotly.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            opacity=0.7,
            name='Objective Function',
            line=dict(color="black")
        ))


        # plotting the guesses
        self.optimisation_path_graph.add_trace(plotly.Scatter(
            x=minimisation_guesses,
            y=y_values_guess,
            mode="markers+lines",
            marker=dict(size=2, color=self.line_colors[1], symbol="circle"),
            line=dict(color=self.line_colors[1], width=2),
            name="(1)",
            opacity=0.5
        ))

        self.optimisation_path_graph.update_layout(
            title='Optimisation Path',
            xaxis_title='x',
            yaxis_title='f(x)',
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            ),
            legend=dict(
            orientation="h",   
            yanchor="bottom",  
            y=1.02,             
            xanchor="right",   
            x=1,
            bgcolor="rgba(255,255,255,0.5)"
        ))

        return self.optimisation_path_graph
    
    def add_optimisation_path(self, minimisation_guesses, objective, exp_number):
        print("!")
        y_values_guess = [objective(torch.tensor(x)) for x in minimisation_guesses]
        print("?")
        self.optimisation_path_graph.add_trace(plotly.Scatter(
            x=minimisation_guesses,
            y=y_values_guess,
            mode="markers+lines",
            marker=dict(size=2, color=self.line_colors[exp_number], symbol="circle"),
            line=dict(color=self.line_colors[exp_number], width=2),
            name=f"({exp_number})",
            opacity=0.5
        ))

        return self.optimisation_path_graph
        
    def update_all_graphs_min(self, minimisation_guesses, gradient_logs, divergence_logs, objective, exp_number):
        updated_optimisation = self.add_optimisation_path(minimisation_guesses=minimisation_guesses, objective=objective, exp_number=exp_number)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        return updated_optimisation, updated_gradient, updated_divergence

    def update_all_graphs_approx(self, loss_logs, gradient_logs, divergence_logs, prediction_data, exp_number):
        updated_loss = self.add_loss_curve(loss_logs=loss_logs, exp_number=exp_number)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_approx = self.add_function_approximation(prediction_data=prediction_data, exp_number=exp_number)
        return updated_loss, updated_gradient, updated_divergence, updated_approx