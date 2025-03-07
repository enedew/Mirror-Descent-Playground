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
        self.trajectories = [] 

    
    def create_loss_curve(self, loss_logs):
        self.loss_curve_graph = plotly.Figure()
        self.loss_curve_graph.add_trace(plotly.Scatter(
            x=list(range(len(loss_logs))),
            y=loss_logs,
            mode="lines",
            opacity=0.7,
            line=dict(color=self.line_colors[1]),
            name="1"
        ))
        self.loss_curve_graph.update_layout(
            title="Loss curve",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            ),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd"
        )
        self.loss_curve_graph.update_yaxes(
            type="log"
        )
        self.loss_curve_graph.update_xaxes(
        gridcolor= "#e8dac5",
        linecolor= "#322634",
        zerolinecolor="#e8dac5"
        )

        self.loss_curve_graph.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )
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
        self.gradient_norm_graph = plotly.Figure()
        self.gradient_norm_graph.add_trace(plotly.Scatter(
            x=list(range(len(gradient_logs))),
            y = gradient_logs,
            mode="lines",
            name="1",
            opacity=0.7,
            line=dict(color=self.line_colors[1]),
        ))
        self.gradient_norm_graph.update_layout(
            title = f"Gradient Norm over {xaxis}s",
            xaxis_title = xaxis,
            yaxis_title = "Gradient Norm",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            ),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd"
        )
        self.gradient_norm_graph.update_xaxes(
        gridcolor= "#e8dac5",
        linecolor= "#322634",
        zerolinecolor="#e8dac5"
        )

        self.gradient_norm_graph.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )
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
        self.divergence_graph = plotly.Figure()
        self.divergence_graph.add_trace(plotly.Scatter(
            x = list(range(len(divergence_logs))),
            y = divergence_logs,
            opacity=0.7,
            mode = "lines",
            name = "1",
            line=dict(color=self.line_colors[1])
        ))
        self.divergence_graph.update_layout(
            title="Bregman Divergence of parameters",
            xaxis_title=f"{xaxis}s",
            yaxis_title="Divergence Value",
            template="plotly_white",
            font=dict(
                family="Roboto",
                color="black"
            ),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd"
        )
        self.divergence_graph.update_xaxes(
        gridcolor= "#e8dac5",
        linecolor= "#322634",
        zerolinecolor="#e8dac5"
        )

        self.divergence_graph.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )
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
        self.approximation_graph = plotly.Figure()
        self.approximation_graph.add_trace(plotly.Scatter(
            x=prediction_data['X'],
            y=prediction_data['Y_true'],
            mode='lines',
            name='True Function',
            line=dict(color="black")
        ))
        self.approximation_graph.add_trace(plotly.Scatter(
            x=prediction_data['X'],
            y=prediction_data['Y_pred'],
            mode='markers+lines',
            name='1',
            opacity=0.5,
            line=dict(color=self.line_colors[1]),
            marker=dict(size=2, opacity=0.5)
        ))
        self.approximation_graph.update_layout(
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
            ),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd"  
        )
        self.approximation_graph.update_xaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"
        )

        self.approximation_graph.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )
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
        
    def create_optimisation_path_graph(self, minimisation_guesses, objective, dim ):
        self.optimisation_path_graph = plotly.Figure()
        self.trajectories.append(minimisation_guesses)
        print(minimisation_guesses[:10])
        
        # generating a line for the objective function

        if dim == 1: 
            y_values_guess = [objective(torch.tensor(x)) for x in minimisation_guesses]
            x_line = np.linspace(min(minimisation_guesses)-5, max(minimisation_guesses)+5, 500)
            y_line = [objective(torch.tensor(x)) for x in x_line]
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                opacity=0.4,
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
                ),
                plot_bgcolor="#f2e9dd",
                paper_bgcolor="#f2e9dd"
            )
        else: 

            x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.2)
            x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
            y_range = np.linspace(y_bounds[0], y_bounds[1])

            X, Y = np.meshgrid(x_range, y_range)

            Z = np.array([[objective(torch.tensor(x), torch.tensor(y)) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])
                
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            
            self.optimisation_path_graph.add_trace(plotly.Contour(
                x=x_range,
                y=y_range,
                z=Z,
                colorscale=[[0,"#f2e9dd"], [1,"#52432f"]],
                contours=dict(showlabels=True, labelfont=dict(size=10)),
                ncontours=40,
                line=dict(color='rgba(82,67,47,0.7)', width=1),
                hoverinfo='none'
            ))

            
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[1], width=1),
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                name="(1)"
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
                ),
                plot_bgcolor="#f2e9dd",
                paper_bgcolor="#f2e9dd"
            )

        self.optimisation_path_graph.update_xaxes(
        gridcolor= "#e8dac5",
        linecolor= "#322634",
        zerolinecolor="#e8dac5"
        )

        self.optimisation_path_graph.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )

        return self.optimisation_path_graph
    
    def add_optimisation_path(self, minimisation_guesses, objective, exp_number, dim ):
        print("!")
        if dim == 1: 
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
        else: 
            self.trajectories.append(minimisation_guesses)
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[exp_number], width=1),
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                name=f"({exp_number})"
            ))
            self.update_contour(objective)


        return self.optimisation_path_graph
    
    def create_dual_space_trajectory_graph(self, dual_logs, objective, dim):
        
        self.dual_space_graph = plotly.Figure()
        
        print("Dual logs (first 10):", dual_logs[:10])
        
        x_vals = [point[0] for point in dual_logs]
        y_vals = [point[1] for point in dual_logs]
        
        
        self.dual_space_graph.add_trace(plotly.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            line=dict(color=self.line_colors[1], width=2),
            marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
            name="Dual Trajectory"
        ))
        
        self.dual_space_graph.update_layout(
            title='Dual Space Optimisation Path',
            xaxis_title='y[0]',
            yaxis_title='y[1]',
            font=dict(family="Roboto", color="black"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd"
        )
        self.dual_space_graph.update_xaxes(
            gridcolor="#e8dac5",
            linecolor="#322634",
            zerolinecolor="#e8dac5"
        )
        self.dual_space_graph.update_yaxes(
            gridcolor="#e8dac5",
            linecolor="#322634",
            zerolinecolor="#e8dac5"
        )
        
        return self.dual_space_graph
    
    def add_dual_space_trajectory(self, dual_logs, exp_number):
        
        
        if isinstance(dual_logs[0], (list, tuple, np.ndarray)) and len(dual_logs[0]) == 2:
            
            x_vals = [point[0] for point in dual_logs]
            y_vals = [point[1] for point in dual_logs]
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[exp_number], width=2),
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                name=exp_number
            ))
            
            new_x_min = min(x_vals) - 1
            new_x_max = max(x_vals) + 1
            new_y_min = min(y_vals) - 1
            new_y_max = max(y_vals) + 1
            
            current_x_range = self.dual_space_graph.layout.xaxis.range if self.dual_space_graph.layout.xaxis.range else [new_x_min, new_x_max]
            current_y_range = self.dual_space_graph.layout.yaxis.range if self.dual_space_graph.layout.yaxis.range else [new_y_min, new_y_max]
            new_x_min = min(new_x_min, current_x_range[0])
            new_x_max = max(new_x_max, current_x_range[1])
            new_y_min = min(new_y_min, current_y_range[0])
            new_y_max = max(new_y_max, current_y_range[1])
            self.dual_space_graph.update_xaxes(range=[new_x_min, new_x_max])
            self.dual_space_graph.update_yaxes(range=[new_y_min, new_y_max])
        else:
            
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=list(range(len(dual_logs))),
                y=dual_logs,
                mode='markers+lines',
                marker=dict(size=2, color=self.line_colors[exp_number], symbol="circle"),
                line=dict(color=self.line_colors[exp_number], width=2),
                name=exp_number,
                opacity=0.5
            ))
            
            new_y_min = min(dual_logs) - 1
            new_y_max = max(dual_logs) + 1
            current_y_range = self.dual_space_graph.layout.yaxis.range if self.dual_space_graph.layout.yaxis.range else [new_y_min, new_y_max]
            new_y_min = min(new_y_min, current_y_range[0])
            new_y_max = max(new_y_max, current_y_range[1])
            self.dual_space_graph.update_yaxes(range=[new_y_min, new_y_max])
        
        return self.dual_space_graph
    
    def create_interactive_bregman_graph(self, x, y):
        def phi(x):
            return x**2
    
        def grad_phi(x):
            return 2*x

        x_vals = np.linspace(-2, 2, 400)
    
        # getting all phi(x) 
        phi_vals = phi(x_vals)

        # taylor expansion at y for all x
        taylor = phi(y) + grad_phi(y) * (x_vals - y)
        
        # calculate values at evaluation point
        phi_x_sel = phi(x)
        taylor_x_sel = phi(y) + grad_phi(y) * (x - y)
        
       
        self.interactive_bregman_plot = plotly.Figure()
        
        # phi(x) plot
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=x_vals, y=phi_vals,
            name=r"$\phi$",
            mode='lines',
            line=dict(color=self.line_colors[2])
            
        ))
        
        # tangent line
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=x_vals, y=taylor,
            name=r"$\nabla\phi(y)$",
            mode='lines',
            line=dict(color=self.line_colors[1])
            
        ))
        
        # marking the expansion point on phi(x)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[y], y=[phi(y)],
            mode='markers',
            name=r"$\phi(y)$",
            marker=dict(color=self.line_colors[1], size=10),
           
        ))
        
        # marking the evaluation point on phi(X)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x], y=[phi_x_sel],
            name=r"$\phi(x)$",
            mode='markers',
            marker=dict(color=self.line_colors[2], size=10),
          
        ))
        
        # corresponding point on the taylor approximation line
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x], y=[taylor_x_sel],
            name=r"$\langle \nabla\phi(y), x-y \rangle$",
            mode='markers',
            marker=dict(color=self.line_colors[3], size=10),
            
        ))
        
        # vertical line connecting the points (the bregman divergence between x and y)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x, x],
            y=[taylor_x_sel, phi_x_sel],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=r'$\mathbb{D}_{\phi}(x, y)$'
        ))
        
        self.interactive_bregman_plot.update_layout(
            xaxis_title='x',
            yaxis_title='Value',
            template='plotly_white',
            font=dict(
                size=10,
                color="#322634"
            ),
            legend=dict(
                font=dict(
                    size=18,
                    color="#322634"
                ),
                entrywidth=150,
                orientation="h",
                y=1.01,
                x=0,
                yanchor="bottom",
                bgcolor="#f2e9dd",

            ),
            plot_bgcolor= "#f2e9dd",
            paper_bgcolor = "#f2e9dd",
            
            margin=dict(t=70, l=50, r=50, b=50)
        )

        self.interactive_bregman_plot.update_xaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"
        )

        self.interactive_bregman_plot.update_yaxes(
            gridcolor= "#e8dac5",
            linecolor= "#322634",
            zerolinecolor="#e8dac5"

        )
        
        return self.interactive_bregman_plot
    
    def compute_dynamic_range(self, trajectories, padding_ratio=0.2, default_range=(-1,1)):
        # function to compute the range of contour values dynamically each time a new trajectory is added
        points = [p for traject in trajectories for p in traject]
        if not points:
            return default_range, default_range
        
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # ensuring non-zero range
        if x_min == x_max:
            x_min, x_max = default_range
        if y_min == y_max:
            y_min, y_max = default_range
        
        padding_x = (x_max - x_min) * padding_ratio
        padding_y = (y_max - y_min) * padding_ratio
        
        return (x_min -padding_x, x_max + padding_x), (y_min - padding_y, y_max + padding_y) 
    
    def update_contour(self, objective):
        # function for dynamically re-computing the plotly contour for additional trajectories
        x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.1)
        x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
        y_range = np.linspace(y_bounds[0], y_bounds[1])

        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([[objective(torch.tensor(x), torch.tensor(y)) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

        self.optimisation_path_graph.data[0].update(x=x_range, y=y_range, z=Z)
        
    def update_all_graphs_min(self, minimisation_guesses, gradient_logs, divergence_logs, dual_logs, objective, exp_number, dim):
        updated_optimisation = self.add_optimisation_path(minimisation_guesses=minimisation_guesses, objective=objective, exp_number=exp_number, dim=dim)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_dual = self.add_dual_space_trajectory(dual_logs=dual_logs, exp_number=exp_number)
        return updated_optimisation, updated_gradient, updated_divergence, updated_dual

    def update_all_graphs_approx(self, loss_logs, gradient_logs, divergence_logs, prediction_data, exp_number):
        updated_loss = self.add_loss_curve(loss_logs=loss_logs, exp_number=exp_number)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_approx = self.add_function_approximation(prediction_data=prediction_data, exp_number=exp_number)
        return updated_loss, updated_gradient, updated_divergence, updated_approx