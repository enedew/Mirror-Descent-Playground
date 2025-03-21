import plotly.graph_objects as plotly 
import numpy as np
import torch 
from scipy.interpolate import griddata
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
            paper_bgcolor="#f2e9dd",
            legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=5, r=10, t=80, b=5)

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
        print("constructed gradient figure")
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
            print("added to gradient figure")
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
            paper_bgcolor="#f2e9dd",
            legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=5, r=10, t=80, b=5)

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
        print("constructed divergence figure")
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
            print("added to divergence figure")
            return self.divergence_graph
        else: 
            raise ValueError("divergence graph does not exist.")
        
    def barycentric_to_cartesian(self,p):
        # p is a list/array of three elements (p1, p2, p3) with p1+p2+p3=1
        # using vertices: a=(0,0), b=(1,0), c=(0.5, sqrt(3)/2)
        x = p[1] + 0.5 * p[2]
        y = (3**0.5)/2 * p[2]
        return x, y


    def create_optimisation_path_graph(self, minimisation_guesses, objective, dim):
        # append trajectory and print first few guesses
        self.trajectories.append(minimisation_guesses)
        print(minimisation_guesses[:10])
        
        if dim == 1:
            # for 1d, compute objective values for each guess
            y_values_guess = [objective(torch.tensor(x)) for x in minimisation_guesses]
            # build frames that add points progressively
            frames = []
            for i in range(1, len(minimisation_guesses) + 1):
                frame = {
                    'data': [dict(
                        x = minimisation_guesses[:i],
                        y = y_values_guess[:i],
                        mode = 'markers+lines',
                        marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                        line = dict(color=self.line_colors[1], width=2),
                        name = "(1)",
                        opacity = 0.5
                    )],
                    'name': str(i)
                }
                frames.append(frame)
            # create figure with frames
            self.optimisation_path_graph = plotly.Figure(frames=frames)
            # add static trace for the objective function curve
            x_line = np.linspace(min(minimisation_guesses) - 5, max(minimisation_guesses) + 5, 500)
            y_line = [objective(torch.tensor(x)) for x in x_line]
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x = x_line,
                y = y_line,
                mode = 'lines',
                opacity = 0.4,
                name = 'objective function',
                line = dict(color="black")
            ))
            # add final (hidden) trace for the full trajectory so layout is defined
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x = minimisation_guesses,
                y = y_values_guess,
                mode = 'markers+lines',
                marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                line = dict(color=self.line_colors[1], width=2),
                name = "(1)",
                opacity = 0.5,
                
            ))
            self.optimisation_path_graph.update_layout(
                title = 'Primal Space Optimisation Trajectory (2D / 1D)',
                xaxis_title = 'x',
                yaxis_title = 'f(x)',
                font = dict(family="roboto", color="black"),
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor = "#f2e9dd",
                paper_bgcolor = "#f2e9dd",
                margin=dict(l=5, r=10, t=80, b=5)
            )
        
        elif dim == 2:
            # for 2d, extract x and y from guesses
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            # build frames that add points progressively
            frames = []
            for i in range(1, len(minimisation_guesses) + 1):
                frame = {
                    'data': [dict(
                        x = x_vals[:i],
                        y = y_vals[:i],
                        mode = 'lines+markers',
                        line = dict(color=self.line_colors[1], width=1),
                        marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                        name = "(1)"
                    )],
                    'name': str(i)
                }
                frames.append(frame)
            self.optimisation_path_graph = plotly.Figure(frames=frames)
            # add static contour plot of the objective function
            x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.2)
            x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
            y_range = np.linspace(y_bounds[0], y_bounds[1], 200)
            X, Y = np.meshgrid(x_range, y_range)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)

            Z_tensor = objective(X_tensor, Y_tensor)
            Z = Z_tensor.detach().numpy()
            
            # if function shoots to infinity at some point (like with the itakura-based objective)
            # then need to clip the Z values or no contours will be visible
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)
            if zmax > 40000:
                threshold = zmin + 0.0001 * (zmax - zmin)
                Z = np.clip(Z, zmin, threshold)

            self.optimisation_path_graph.add_trace(plotly.Contour(
                x = x_range,
                y = y_range,
                z = Z,
                colorbar = dict(title="f(x)"),
                colorscale = [[0, "#f2e9dd"], [1, "#52432f"]],
                contours = dict(showlabels=True, labelfont=dict(size=10)),
                ncontours = 40,
                line = dict(color='rgba(82,67,47,0.7)', width=1),
                hoverinfo = 'none'
            ))
            # add final (hidden) trace for the full trajectory so layout is defined
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x = x_vals,
                y = y_vals,
                mode = 'lines+markers',
                line = dict(color=self.line_colors[1], width=1),
                marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                name = "(1)",
            ))
            self.optimisation_path_graph.update_layout(
                title = 'Primal Space Optimisation Trajectory (2D / 1D)',
                xaxis_title = 'x',
                yaxis_title = 'y',
                template = "plotly_white",
                font = dict(family="roboto", color="black"),
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor = "#f2e9dd",
                paper_bgcolor = "#f2e9dd",
                margin=dict(l=5, r=10, t=80, b=5)
            )
        
            
        
        elif dim == 3:
            # for simplex problems (3d), first create a static contour
            # make u,v axes and grid
            u = np.linspace(0, 1, 100)
            v = np.linspace(0, 1, 100)
            U, V = np.meshgrid(u, v)

            # only keep points lying on the simplex
            mask = U + V <= 1

            #barycentric coordinates p1, p2, p3
            p1 = U[mask]
            p2 = V[mask]
            p3 = 1 - p1 - p2

            # convert to cartesian coordinates
            points_cart = np.array([self.barycentric_to_cartesian([p1_val, p2_val, p3_val])
                                    for p1_val, p2_val, p3_val in zip(p1, p2, p3)])
            # get x and y values
            xi = points_cart[:, 0]
            yi = points_cart[:, 1]

            # create the meshgrid for contour and then go through the objective values
            z_scatter = np.array([objective(torch.tensor([p1_val, p2_val, p3_val], dtype=torch.float32)).item()
                                for p1_val, p2_val, p3_val in zip(p1, p2, p3)])
            
            x_grid = np.linspace(xi.min(), xi.max(), 200)
            y_grid = np.linspace(yi.min(), yi.max(), 200)
            XI, YI = np.meshgrid(x_grid, y_grid)

            ZI = griddata((xi, yi), z_scatter, (XI, YI), method='linear')
            # filter out nan values
            valid_Z = ZI[~np.isnan(ZI)]

            # clip any extreme values as the simplex objective has extremely high values around the edge of the simplex
            zmin = valid_Z.min()
            zmax = valid_Z.max()
            threshold = zmin + 0.2 * (zmax - zmin)
            ZI_clipped = np.clip(ZI, zmin, threshold)

            # set the number of contours and add the contour trace
            n_contours = 20
            contour_step = (threshold - zmin) / n_contours
            self.optimisation_path_graph = plotly.Figure()
            self.optimisation_path_graph.add_trace(plotly.Contour(
                x = x_grid,
                y = y_grid,
                z = ZI_clipped,
                colorscale = [[0, "#f2e9dd"], [1, "#52432f"]],
                contours = dict(
                    showlabels=True,
                    labelfont=dict(size=10),
                    start=zmin,
                    end=threshold,
                    size=contour_step
                ),
                line = dict(color='rgba(82,67,47,0.7)', width=1),
                hoverinfo = 'none'
            ))
            # build frames for the trajectory scatter trace
            traj_cart = [self.barycentric_to_cartesian(p) for p in minimisation_guesses]
            traj_x = [p[0] for p in traj_cart]
            traj_y = [p[1] for p in traj_cart]
            hover_text = [f"p1={p[0]:.2f}, p2={p[1]:.2f}, p3={p[2]:.2f}" for p in minimisation_guesses]
            frames = []
            for i in range(1, len(traj_x) + 1):
                frame = {
                    'data': [dict(
                        x = traj_x[:i],
                        y = traj_y[:i],
                        mode = 'lines+markers',
                        line = dict(color=self.line_colors[1], width=1),
                        marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                        name = "(1)",
                        hovertext = hover_text[:i],
                        hovertemplate = "%{hovertext}"
                    )],
                    'name': str(i)
                }
                frames.append(frame)
            self.optimisation_path_graph.frames = frames
            # add final (hidden) trace for the full trajectory
            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x = traj_x,
                y = traj_y,
                mode = 'lines+markers',
                line = dict(color=self.line_colors[1], width=1),
                marker = dict(size=2, color=self.line_colors[1], symbol='circle'),
                name = "(1)",
                hovertext = hover_text,
                hovertemplate = "%{hovertext}",
                
            ))
            # add annotations for triangle vertices
            vertices = {
                "p=(1,0,0)": self.barycentric_to_cartesian([1, 0, 0]),
                "p=(0,1,0)": self.barycentric_to_cartesian([0, 1, 0]),
                "p=(0,0,1)": self.barycentric_to_cartesian([0, 0, 1])
            }
            annotations = []
            for text, (x_val, y_val) in vertices.items():
                annotations.append(dict(
                    x = x_val,
                    y = y_val,
                    xref = "x",
                    yref = "y",
                    text = text,
                    showarrow = True,
                    arrowhead = 2,
                    ax = 0,
                    ay = -20,
                    font = dict(color="black", size=12)
                ))
            self.optimisation_path_graph.update_layout(
                title = 'Primal Space Optimisation Trajectory (2D / 1D)',
                xaxis_title = '',
                yaxis_title = '',
                template = "plotly_white",
                font = dict(family="roboto", color="black"),
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor = "#f2e9dd",
                paper_bgcolor = "#f2e9dd",
                annotations = annotations,
                margin=dict(l=5, r=10, t=80, b=5)
            )
            self.optimisation_path_graph.update_xaxes(
                gridcolor = "#e8dac5",
                linecolor = "#322634",
                zerolinecolor = "#e8dac5"
            )
            self.optimisation_path_graph.update_yaxes(
                gridcolor = "#e8dac5",
                linecolor = "#322634",
                zerolinecolor = "#e8dac5"
            )
        print("Constructed 2D contour figure")
        return self.optimisation_path_graph
    
    def add_optimisation_path(self, minimisation_guesses, objective, exp_number, dim):
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
        elif dim == 2:
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
        elif dim == 3:
            # for simplex, convert barycentrics to cartesian
            def bary_to_cart(p):
                return self.barycentric_to_cartesian(p)
            self.trajectories.append(minimisation_guesses)
            traj_cart = [bary_to_cart(p) for p in minimisation_guesses]
            x_vals = [pt[0] for pt in traj_cart]
            y_vals = [pt[1] for pt in traj_cart]
            hover_text = [f"p1={p[0]:.2f}, p2={p[1]:.2f}, p3={p[2]:.2f}" for p in minimisation_guesses]

            self.optimisation_path_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[exp_number], width=1),
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                name=f"({exp_number})",
                hovertext=hover_text,
                hovertemplate="%{hovertext}"
            ))
            # update the contour plot for the simplex if needed
        print("Added to 2D contour figure")
        return self.optimisation_path_graph

    def create_optimisation_path_3d_graph(self, minimisation_guesses, objective, dim):
        # create a 3d graph for the optimisation trajectory

        if dim == 1: 
            return None
        if dim == 2:
            # for 2d problems, compute x, y and objective value as z
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            z_vals = [objective(*torch.tensor([p[0], p[1]], dtype=torch.float32)).item() for p in minimisation_guesses]
            # create grid for the objective surface
            
            x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.2)
            x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
            y_range = np.linspace(y_bounds[0], y_bounds[1], 200)
            X, Y = np.meshgrid(x_range, y_range)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)

            Z_tensor = objective(X_tensor, Y_tensor)
            Z = Z_tensor.detach().numpy()
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)
            if zmax > 40000:
                threshold = zmin + 0.0001 * (zmax - zmin)
                Z = np.clip(Z, zmin, threshold)

            # create the figure and add the surface and trajectory traces
            self.optimisation_path_graph3d = plotly.Figure()
            self.optimisation_path_graph3d.add_trace(plotly.Surface(
                x=X,
                y=Y,
                z=Z,
                opacity=0.4,
                name='objective surface'
            ))
            self.optimisation_path_graph3d.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                line=dict(color=self.line_colors[1], width=2),
                name="(1)"
            ))
            self.optimisation_path_graph3d.update_layout(
                title='Primal Space Optimisation Trajectory (3D)',
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='f(x,y)'
                ),
                font=dict(family="roboto", color="black"),
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="#f2e9dd",
                paper_bgcolor="#f2e9dd",
                margin=dict(l=5, r=10, t=80, b=5),
                transition={
                    'duration': 500,
                    'easing': 'cubic-in-out'
                }
            )
        elif dim == 3:
            # for 3d problems, assume minimisation_guesses is a list of 3d points
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            z_vals = [p[2] for p in minimisation_guesses]
            self.optimisation_path_graph3d = plotly.Figure()
            self.optimisation_path_graph3d.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                line=dict(color=self.line_colors[1], width=2),
                name="(1)"
            ))
            self.optimisation_path_graph3d.update_layout(
                title='Primal Space Optimisation Trajectory (3D)',
                scene=dict(
                    xaxis_title='p1',
                    yaxis_title='p2',
                    zaxis_title='p3'
                ),
                font=dict(family="roboto", color="black"),
                plot_bgcolor="#f2e9dd",
                paper_bgcolor="#f2e9dd",
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=5, r=10, t=80, b=5),
                transition={
                    'duration': 500,
                    'easing': 'cubic-in-out'
                }

            )
        print("3D figure constructed")
        return self.optimisation_path_graph3d

    def add_optimisation_path_3d(self, minimisation_guesses, objective, exp_number, dim):
        if dim == 2:
            # add the new trajectory to stored trajectories
            if not hasattr(self, 'trajectories'):
                self.trajectories = []
            self.trajectories.append(minimisation_guesses)

            # compute x, y, z for new trajectory
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            z_vals = [objective(*torch.tensor([p[0], p[1]], dtype=torch.float32)).item() for p in minimisation_guesses]

            # recompute the global bounds for all trajectories
            all_points = [pt for traj in self.trajectories for pt in traj]
            x_all = [pt[0] for pt in all_points]
            y_all = [pt[1] for pt in all_points]
            x_min, x_max = min(x_all), max(x_all)
            y_min, y_max = min(y_all), max(y_all)

            # create a new grid of values
            x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.2)
            x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
            y_range = np.linspace(y_bounds[0], y_bounds[1], 200)
            X, Y = np.meshgrid(x_range, y_range)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)

            Z_tensor = objective(X_tensor, Y_tensor)
            Z = Z_tensor.detach().numpy()
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)
            if zmax > 40000:
                threshold = zmin + 0.0001 * (zmax - zmin)
                Z = np.clip(Z, zmin, threshold)

            # update the existing surface to account for new trajectory
            self.optimisation_path_graph3d.data[0].update(x=X, y=Y, z=Z)

            # add the new trajectory 
            self.optimisation_path_graph3d.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                line=dict(color=self.line_colors[exp_number], width=2),
                name=f"({exp_number})"
            ))
        elif dim == 3:
            # for 3d (simplex) problems, no surface needs to be updated
            x_vals = [p[0] for p in minimisation_guesses]
            y_vals = [p[1] for p in minimisation_guesses]
            z_vals = [p[2] for p in minimisation_guesses]
            self.optimisation_path_graph3d.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                line=dict(color=self.line_colors[exp_number], width=2),
                name=f"({exp_number})"
            ))

        elif dim == 1:
            return None
        print("added to 3d figure")
        return self.optimisation_path_graph3d
    def create_dual_space_trajectory_graph(self, dual_logs, objective, dim):
        self.dual_space_graph = plotly.Figure()
        
        if dim == 1:
            x_vals = [point for point in dual_logs]
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=dual_logs,
                mode='lines+markers',
                line=dict(color=self.line_colors[1], width=2),
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                name="(1)"
            ))
        elif dim == 2:
            x_vals = [point[0] for point in dual_logs]
            y_vals = [point[1] for point in dual_logs]
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[1], width=2),
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                name="(1)"
            ))
        elif dim == 3:
            # for simplex problems im using a 3d scatter plot
            # tried a ternary diagram, but dual space values can be negative for some divergences
            # so in that case the trajectory isnt displayed properly
            x_vals = [p[0] for p in dual_logs]
            y_vals = [p[1] for p in dual_logs]
            z_vals = [p[2] for p in dual_logs]
            hover_text = [f"p1-dual={p[0]:.2f}<br>p2-dual={p[1]:.2f}<br>p3-dual={p[2]:.2f}" for p in dual_logs]
            
            self.dual_space_graph.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=2, color=self.line_colors[1], symbol='circle'),
                line=dict(color=self.line_colors[1], width=2),
                name="(1)",
                hovertext=hover_text,
                hovertemplate="%{hovertext}"
            ))
            
            self.dual_space_graph.update_layout(
                title='Dual Space Optimisation Trajectory',
                scene=dict(
                    xaxis_title='p1',
                    yaxis_title='p2',
                    zaxis_title='p3',
                    bgcolor="#f2e9dd"
                ),
                paper_bgcolor="#f2e9dd",
                font=dict(family="roboto", color="black"),
                legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=5, r=10, t=80, b=5)
            )

            return self.dual_space_graph
        self.dual_space_graph.update_layout(
            title='Dual Space Optimisation Trajectory',
            xaxis_title='x',
            yaxis_title='y',
            font=dict(family="roboto", color="black"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#f2e9dd",
            paper_bgcolor="#f2e9dd",
            margin=dict(l=5, r=10, t=80, b=5)
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
        print("constructed dual space figure")
        return self.dual_space_graph
    
    def add_dual_space_trajectory(self, dual_logs, exp_number, dim):
        if dim == 1:
            x_vals = [point for point in dual_logs]
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=dual_logs,
                mode='markers+lines',
                marker=dict(size=2, color=self.line_colors[exp_number], symbol="circle"),
                line=dict(color=self.line_colors[exp_number], width=2),
                name=str(exp_number)
            ))
        elif dim == 2:
            x_vals = [point[0] for point in dual_logs]
            y_vals = [point[1] for point in dual_logs]
            self.dual_space_graph.add_trace(plotly.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=self.line_colors[exp_number], width=2),
                marker=dict(size=2, color=self.line_colors[exp_number], symbol='circle'),
                name=str(exp_number)
            ))
        elif dim == 3:
            x_vals = [p[0] for p in dual_logs]
            y_vals = [p[1] for p in dual_logs]
            z_vals = [p[2] for p in dual_logs]
            
            # create hover text to display the original barycentrics
            hover_text = [f"p1-dual={p[0]:.2f}<br>p2-dual={p[1]:.2f}<br>p3-dual={p[2]:.2f}" for p in dual_logs]
            
            self.dual_space_graph.add_trace(plotly.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='lines+markers',
                marker=dict(size=3, color=self.line_colors[exp_number % len(self.line_colors)], symbol='circle'),
                line=dict(color=self.line_colors[exp_number % len(self.line_colors)], width=2),
                name=f"({exp_number})",
                hovertext=hover_text,
                hovertemplate="%{hovertext}",
                visible=True
            ))
            
    
        print("added to dual space figure")
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
            x_min, x_max = x_min - 1, x_max + 1
        if y_min == y_max:
            y_min, y_max = y_min - 1, y_max + 1
        
        padding_x = (x_max - x_min) * padding_ratio
        padding_y = (y_max - y_min) * padding_ratio
        
        return (x_min -padding_x, x_max + padding_x), (y_min - padding_y, y_max + padding_y) 
    
    def update_contour(self, objective):
        # function for dynamically re-computing the plotly contour for additional trajectories
        x_bounds, y_bounds = self.compute_dynamic_range(self.trajectories, padding_ratio=0.1)
        x_range = np.linspace(x_bounds[0], x_bounds[1], 200)
        y_range = np.linspace(y_bounds[0], y_bounds[1])

        X, Y = np.meshgrid(x_range, y_range)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        Z_tensor = objective(X_tensor, Y_tensor)
        Z = Z_tensor.detach().numpy()
        zmin = np.nanmin(Z)
        zmax = np.nanmax(Z)
        if zmax > 40000:
            threshold = zmin + 0.0001 * (zmax - zmin)
            Z = np.clip(Z, zmin, threshold)


        self.optimisation_path_graph.data[0].update(x=x_range, y=y_range, z=Z)
        
    def update_all_graphs_min(self, minimisation_guesses, gradient_logs, divergence_logs, dual_logs, objective, exp_number, dim):
        updated_optimisation = self.add_optimisation_path(minimisation_guesses=minimisation_guesses, objective=objective, exp_number=exp_number, dim=dim)
        updated_optimisation_3d = self.add_optimisation_path_3d(minimisation_guesses=minimisation_guesses, objective=objective, exp_number=exp_number, dim=dim)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_dual = self.add_dual_space_trajectory(dual_logs=dual_logs, exp_number=exp_number, dim=dim)
        return (updated_optimisation, updated_optimisation_3d, updated_gradient, updated_divergence, updated_dual)

    def update_all_graphs_approx(self, loss_logs, gradient_logs, divergence_logs, prediction_data, exp_number):
        updated_loss = self.add_loss_curve(loss_logs=loss_logs, exp_number=exp_number)
        updated_gradient =self.add_gradient_norm(gradient_logs=gradient_logs, exp_number=exp_number)
        updated_divergence = self.add_divergence(divergence_logs=divergence_logs, exp_number=exp_number)
        updated_approx = self.add_function_approximation(prediction_data=prediction_data, exp_number=exp_number)
        return updated_loss, updated_gradient, updated_divergence, updated_approx