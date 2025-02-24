import dash
from dash import dcc, html
import plotly.express as px

dash.register_page(__name__, path="/")

mirror_descent_markdown = r"""
### What is Mirror Descent?
**Mirror Descent** is an optimisation algorithm that extends gradient descent to non-Euclidean geometries by utilising different **distance-generating functions**, each of which induce their own **Bregman Divergence**.

The Mirror Descent update rule can be formulated as:
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
### The Algorithm 
Given a convex function $f$ to optimise, a learning rate $\alpha_k$, initial point $x_0$, and 
a distance-generating function $\phi$, with $\nabla\phi$ as the mirror map
Starting from initial $x_0$, in each iteration: 
* Map to the dual space: y_k \from \nabla

"""


layout = html.Div([
    dcc.Markdown(mirror_descent_markdown, mathjax=True, className="markdown-desc"),
    dcc.Markdown(algo_markdown, mathjax=True, className="markdown-algo")
], className="info-page-layout")

