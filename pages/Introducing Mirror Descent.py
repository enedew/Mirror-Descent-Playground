import dash
from Graphs import Graphs
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import numpy as np
dash.register_page(__name__, path="/")

mirror_descent_markdown = r"""
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
Bregman Divergences are a family of distance measuring functions, defined in terms of a strictly convex and differentiable function. The most basic, and most well-known example
of a Bregman Divergence is the squared Euclidean distance. When the chosen bregman divergence for mirror descent is given as the Euclidean distance, this yields the standard
Gradient Descent algorithm.

Bregman Divergences are crucial to understanding the intracacies of the mirror descent algorithm, and how it can adapt to different geometries through a suitable choice of Bregman Divergence.

Let $\phi: \Omega \to \mathbb{R}$ be a continuously differentiable, strictly convex function defined on a convex set $\Omega$.

The Bregman divergence associated with $\phi$ between two points $x, y \in \Omega$ is the difference between the value of $\phi$ at point x and the value of the first-order Taylor exapansion of $\phi$ around point $y$ evaluated at point $x$.

$$
D_\phi(x, y) = \phi(x) - \phi(y) - \langle \nabla\phi(y), x-y \rangle
$$
"""

mirrormap_md = r"""
#### *Quadratic Mirror Map (Euclidean / Mahalanobis)*
**Distance-Generating Function:** $\phi(x) = \frac{1}{2}x^TQx$. This is a convex quadratic form, taking $Q = I$ gives $\frac{1}{2}\lVert{x}\rVert^2_2$ 
which is the Squared Euclidean Norm. Here Q is defined as symmetric positive-definite.

**Mirror Map:** $\nabla\phi(x_k) = Qx_k = y_k$. In the Euclidean case $Q = I$, this is simply $\nabla\phi(x) = x$ - the identity mapping. In the Mahalanobis case,
coordinates are scaled by Q.  


**Inverse Mirror Map:** $(\nabla\phi)^{-1}(y_{k+1}) = Q^{-1}y_{k+1} = x_{k+1}$. For the Euclidean case this is again the identity. In the mahalanobis case the coordinates are scaled by $Q^{-1}$.
  
**Bregman Divergence:** Squared Mahalanobis distance: $D_{\phi}(x, y) = \frac{1}{2}(x - y)^TQ(x-y)$ and in the case where $Q = I$, the Squared Euclidean distance: $D_{\phi}(x, y) = \frac{1}{2}\lVert x - y\rVert^2$

**Notes & Applications:** In the Euclidean case, mirror descent with this mirror map reduces to ordinary gradient descent, 
the default and industry standard optimisation algorithm widely used for its effectiveness and simplicity. The Mahalanobis variant allows for 
weighting or scaling of certain dimensions. This is particularly useful when there is specific knowledge of how the curvature of the objective function varies.
This is easy to visualise with the experiment tool, selecting the anisotropic function preset:  $$a(x - x^*)^2 + b(y - y^*)^2$$ and defining 
$$Q = \begin{bmatrix} a & 0 \\ 0 & b\end{bmatrix}$$. The standard Euclidean mirror map will embark on a curved trajectory according to the varying curvature of the objective, while the better informed Mahalanobis mirror map will embark on a straight, direct path to the optimum due to the information Q provides for the update rule.

---

#### *Negative Entropy Mirror Map (Kullback-Leibler Divergence)*
**Distance-Generating Function:** $\phi(x) =  \sum_{i=1}^n \left[x_i \ln x_i - x_i \right]$, this is a strictly convex function known as negative Shannon entropy, and is typically defined on the probability simplex.

**Mirror Map:** $\nabla\phi(x_k) = [\ln x_{k_1}, \ln x_{k_2}, ... , \ln x_{k_n}] = y_k$. In other words, the mirror map sends $x$ to its natural logarithm in an elementwise manner. 

**Inverse Mirror Map:** $(\nabla\phi)^{-1}(y_{k+1}) = [e^{y_{{k+1}_1}}, e^{y_{{k+1}_2}}, ... , e^{y_{{k+1}_n}}]$. In the unconstrained form, this is an elementwise exponential mapping. Therefore when constrained to the simplex
and normalised for probabilities, this mapping is equivalent to the softmax function.

**Bregman Divergence:** Generalised Kullback-Leibler (KL) Divergence: $D_{KL}(P || Q) = \sum_i \left[P(i)\ln\frac{P(i)}{Q(i)} - P(i) + Q(i)\right]$. When P and Q are probability distributions this reduces to the standard KL Divergence: $\Sigma_i P(i)\ln\frac{P(i)}{Q(i)}$

**Notes & Applications:** This variant of mirror descent is also known as the Exponentiated Gradient algorithm, and is most commonly used for objectives constrained to the simplex. 
The key aspect that sets this mirror map apart from the Euclidean/Mahalanobis mirror maps is that parameters are updated multiplicatively rather than additively, this can be shown by combining the mappings into one step to yield $x_{k+1} = e^{ln_{x_k} - \alpha g_k} = x_k \odot e^{-\alpha g_k}$. It is widely used in machine learning for online learning and boosting, as well as in reinforcement learning for policy optimisation.
The performance benefits of this variant on the simplex can be visualised with the experiment tool by selecting the "3D Simplex" objective preset, and comparing with the other mirror maps.

---

#### *Itakura-Saito Mirror Map*
**Distance-Generating Function:** $\phi(x) = - \sum^n_{i=1} \ln x_i$. This is known as a log-barrier or Burg's entropy for the positive orthant. It is strictly convex for $x_i > 0$.

**Mirror Map:** $\nabla\phi(x_k) = [-\frac{1}{x_{k_1}}, -\frac{1}{x_{k_2}}, ... , -\frac{1}{x_{k_n}}]$. This maps each coordinate for to its negative reciprocal.

**Inverse Mirror Map:** $(\nabla\phi)^{-1}(y_{k+1}) = [-\frac{1}{y_{{k+1}_1}}, -\frac{1}{y_{{k+1}_2}}, ... , -\frac{1}{y_{{k+1}_n}}]$. This again maps each coordinate to its negative reciprocal.

**Bregman Divergence:** Itakura-Saito Divergence: $D_{\phi}(x,y) = \sum^n_{i=1} (\frac{x_i}{y_i} - \ln\frac{x_i}{y_i} - 1)$

**Notes & Applications:** In machine learning, Itakura-Saito divergence is most commonly used in non-negative matrix factorisation as a measure of the quality of factorisation, yet its behaviour as a mirror map is insightful for learning how the geometries of different mirror maps induce widely varied behaviour during optimisation.
It is scale invariant - it measures distance relatively, rather than an absolute measure, such that it remains unchanged if all inputs are scaled by the same positive factor. Parameters are updated multiplicatively, as with the negative entropy mirror map, and its update can be rewritten as $x_{k+1} = \frac{x_k}{1 + \alpha x_k g_k}$.
This leads to updates being incredibly sensitive to large gradients which in turn causes instability, even more so than the negative entropy mirror map, due to the denominator approaching 0. This mirror map performs best in very specific scenarios, such as when the initial point and optimum both lie in a flat valley, as can be seen with the "Itakura-based" preset objective function.
"""


maps_desc = html.Div([
    html.H3("Mirror Maps used and their properties"),

    dcc.Markdown(mirrormap_md, className="padding-markdown", mathjax=True)

], className="maps-desc")


layout = html.Div([
    # first row div - description of the mirror descent algorithm 
    html.Div([
        html.H3("What is Mirror Descent?"),
        dcc.Markdown(mirror_descent_markdown, mathjax=True,className="padding-markdown")
    ], className="mirror-desc"),

    # second row div - two columns for explaining bregmans and a visualisation
    html.Div([
        # first column, description and sliders
        html.Div([
            html.H3("What are Bregman Divergences?"),
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

        # second column - graph visualisation of bregmans
        html.Div([
            dcc.Markdown("### Visualising the Bregman Divergence for the Euclidean norm $\phi=||x||^2$ (squared Euclidean distance)", mathjax=True, className="graph-header"),
            dcc.Graph(
                id="interactive-bregman-fig",
                config={'responsive': True},
                className="graph",
                mathjax=True
            )
        ], className="bregman-graph-interactive")
    ], className="bregman-visual"),
    maps_desc


], className="info-page-layout")

# callback updates the bregman divergence visualisation
@callback(
    Output("interactive-bregman-fig", "figure"), 
    Input("x-value", "value"),
    Input("y-value", "value")
)
def update_bregman_graph(x_value, y_value):
    graph = Graphs()
    fig = graph.create_interactive_bregman_graph(x_value, y_value)
    
    return fig