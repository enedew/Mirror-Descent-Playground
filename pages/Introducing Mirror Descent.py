import dash
from Graphs import Graphs
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import numpy as np
dash.register_page(__name__, path="/")

mirror_descent_markdown = r"""
### What is Mirror Descent?
**Mirror Descent** is an optimisation algorithm that extends gradient descent to non-Euclidean geometries by utilising different **distance-generating functions**, each of which induce their own **Bregman Divergence**.

Say we are trying to minimise some convex function. We have a point $x_k$, the subgradient at this point $g_k$,
and a step size or learning rate $\alpha_k$.
To determine $x_{k+1}$ the Mirror Descent update rule can be formulated as:
$$
x_{k+1} = \arg\min_{x \in X} \left \{\langle g_k, x- x_k \rangle + \frac{1}{\alpha_k}D_\phi(x, x_k) \right \}
$$
where $D_\phi$ is a Bregman Divergence of the form:
$$
D_\phi(x, y) = \phi(x) - \phi(y) - \langle \nabla\phi(y), x-y \rangle
$$
where $\phi$ is a strictly convex, differentiable function known as the distance-generating function.

Say we are trying to minimise some function $f(x)$. The algorithm for mirror descent works by utilising $\nabla\phi$, known as the **Mirror Map** to map values $x_k$ from the primal space to the dual space.
$$
y_k = \nabla\phi(x_k)
$$
The update to $x_k$ is then performed in the dual space as you would in the Gradient Descent algorithm, by performing a step in the negative direction of the subgradient $g_k = \nabla f(x_k)$, scaled by some value $\alpha$ (the learning rate). 
$$
y_{k+1} = y_k - \alpha_k g_k
$$
Now we have performed the update, we can map $y_{k+1}$ back to the primal space using the function $\nabla\phi^*$, where $\phi^*$ is the convex conjugate of $\phi$.
$$
x_{k+1} = \nabla\phi^*(y_{k+1})
$$
"""

algo_markdown = r"""
### What are Bregman Divergences? 
Bregman Divergences are a family of distance measuring functions, defined in terms of a strictly convex and differentiable function. The most basic, and most well-known example
of a Bregman Divergence is the squared Euclidean distance. Indeed, when the chosen bregman divergence for mirror descent is given as the Euclidean distance, this yields the standard
Gradient Descent algorithm.

Bregman Divergences are crucial to understanding the intracacies of the mirror descent algorithm, and how it can adapt to different geometries through a suitable choice of Bregman Divergence.

Let $\phi: \Omega \to \mathbb{R}$ be a continuously differentiable, strictly convex function defined on a convex set $\Omega$.

The Bregman divergence associated with $\phi$ between two points $x, y \in \Omega$ is the difference between the value of $\phi$ at point x and the value of the first-order Taylor exapansion of $\phi$ around point $y$ evaluated at point $x$.

$$
D_\phi(x, y) = \phi(x) - \phi(y) - \langle \nabla\phi(y), x-y \rangle
$$
"""


layout = html.Div([
    # First row: full-width markdown for mirror descent
    html.Div([
        dcc.Markdown(mirror_descent_markdown, mathjax=True,className="padding-markdown")
    ], className="mirror-desc"),

    # Second row: two equally sized columns
    html.Div([
        # Left column: Markdown algo_markdown and sliders
        html.Div([
            dcc.Markdown(algo_markdown, mathjax=True, className="padding-markdown"),
            html.Div([
                html.Div([
                    dcc.Markdown("$x$", mathjax=True, className="bregman-example-inputs"),
                    dcc.Slider(
                        id="x-value",
                        min=-3,
                        max=3,
                        step=0.1,
                        value=-1.5,
                        marks={i: str(i) for i in np.arange(-2, 2.1, 0.5)}
                    )
                ], className="bregman-slider"),
                html.Div([
                    dcc.Markdown("$y$", mathjax=True, className="bregman-example-inputs"),
                    dcc.Slider(
                        id="y-value",
                        min=-3,
                        max=3,
                        step=0.1,
                        value=1,
                        marks={i: str(i) for i in np.arange(-2, 2.1, 0.5)}
                    )
                ], className="bregman-slider")
            ], className="bregman-sliders")
        ], className="bregman-desc"),

        # Right column: the graph
        html.Div([
            dcc.Markdown("### Visualising the Bregman Divergence for the Euclidean norm $\phi=||x||^2$ (squared Euclidean distance)", mathjax=True, className="padding-markdown"),
            dcc.Graph(
                id="interactive-bregman-fig",
                config={'responsive': True},
                className="graph",
                mathjax=True
            )
        ], className="bregman-graph-interactive")
    ], className="bregman-visual")
], className="info-page-layout")

@callback(
    Output("interactive-bregman-fig", "figure"), 
    Input("x-value", "value"),
    Input("y-value", "value")
)
def update_bregman_graph(x_value, y_value):
    graph = Graphs()
    fig = graph.create_interactive_bregman_graph(x_value, y_value)
    
    return fig