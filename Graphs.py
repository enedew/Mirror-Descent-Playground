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
                colorscale='greys',
                contours=dict(showlabels=True, labelfont=dict(size=10)),
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
    
    def create_interactive_bregman_graph(self, x, y):
        def phi(x):
            return x**2
    
        def grad_phi(x):
            return 2*x

        # Create x values for plotting
        x_vals = np.linspace(-2, 2, 400)
    
        # Calculate phi(x) for all x
        phi_vals = phi(x_vals)
        # Calculate the Taylor expansion at y for all x: T(x; y)=phi(y)+phi'(y)(x-y)
        taylor = phi(y) + grad_phi(y) * (x_vals - y)
        
        # Calculate the values at the selected evaluation point x_sel
        phi_x_sel = phi(x)
        taylor_x_sel = phi(y) + grad_phi(y) * (x - y)
        
        # Create the Plotly figure and add traces
        self.interactive_bregman_plot = plotly.Figure()
        
        # Plot phi(x)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=x_vals, y=phi_vals,
            name=r"$\phi$",
            mode='lines',
            line=dict(color=self.line_colors[2])
            
        ))
        
        # Plot the tangent line (Taylor expansion at y)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=x_vals, y=taylor,
            name=r"$\nabla\phi(y)$",
            mode='lines',
            line=dict(color=self.line_colors[1])
            
        ))
        
        # Mark the expansion point on phi(x)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[y], y=[phi(y)],
            mode='markers',
            name=r"$\phi(y)$",
            marker=dict(color=self.line_colors[1], size=10),
           
        ))
        
        # Mark the selected evaluation point on phi(x)
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x], y=[phi_x_sel],
            name=r"$\phi(x)$",
            mode='markers',
            marker=dict(color=self.line_colors[2], size=10),
          
        ))
        
        # Mark the corresponding point on the tangent line
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x], y=[taylor_x_sel],
            name=r"$\langle \nabla\phi(y), x-y \rangle$",
            mode='markers',
            marker=dict(color=self.line_colors[3], size=10),
            
        ))
        
        # Draw a vertical line connecting the function value and its tangent at x_sel
        self.interactive_bregman_plot.add_trace(plotly.Scatter(
            x=[x, x],
            y=[taylor_x_sel, phi_x_sel],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=r'$\mathbb{D}_{\phi}(x, y)$'
        ))
        
        # Update the layout of the figure
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
        x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.2)
        x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
        y_range = np.linspace(y_bounds[0], y_bounds[1])

        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([[objective(torch.tensor(x), torch.tensor(y)) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

        self.optimisation_path_graph.data[0].update(x=x_range, y=y_range, z=Z)
        
    def update_all_graphs_min(self, minimisation_guesses, gradient_logs, divergence_logs, objective, exp_number, dim):
        updated_optimisation = self.add_optimisation_path(minimisation_guesses=minimisation_guesses, objective=objective, exp_number=exp_number, dim=dim)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        return updated_optimisation, updated_gradient, updated_divergence

    def update_all_graphs_approx(self, loss_logs, gradient_logs, divergence_logs, prediction_data, exp_number):
        updated_loss = self.add_loss_curve(loss_logs=loss_logs, exp_number=exp_number)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_approx = self.add_function_approximation(prediction_data=prediction_data, exp_number=exp_number)
        return updated_loss, updated_gradient, updated_divergence, updated_approx