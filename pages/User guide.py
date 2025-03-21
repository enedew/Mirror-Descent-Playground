import dash
from dash import html, dcc 

dash.register_page(__name__, path="/guide")
mirrormap_md = r"""
#### *Configuration*
The experiment tool allows for up to 5 Mirror Descent configurations to be ran on the same objective, with results overlaid on displayed figures for direct comparison. Users can add/remove configurations using the buttons at the bottom of the Configuration panel.

* **Objective Function:** You can input your own custom objective function, using standard python syntax for mathematical expressions. The parser allows for only these functions and constants with the following syntax when used in conjunction with another variable $x$:
    * $\sin$: sin(x)
    * $\cos$: cos(x)
    * $\tan$: tan(x)
    * $\exp$: exp(x) 
    * $\log$: log(x)
    * $\pi$: pi*x

You can also choose from one of the available objective presets, which allow for customising curvature and where the optimum lies. The key distinction between inputting a custom function and choosing from a preset, is that all presets have known optimums and so converge iteration and distance from optimum can be measured after the experiment has ran.

* **Parameters (Objective)**
    * Depending on the chosen objective presets, different parameters will be shown / hidden accordingly. 
    * Some allow for scaling variables $a$ or $b$, or setting target weights or coordinates of the optimum
    * All allow for a noise input - this uses the cosine function to add a perturbation to the 2D surface, making the optimisation process more difficult. It fades once close to the optimum to ensure the optimum is not changed by the noise.

* **Parameters (per Configuration)**
    * *Initial value* - Specifies the initial point which the mirror descent algorithm will start the optimisation process from. Please note for simplex problems, if the initial values or target values are not normalised this may cause irregular behaviour.
    * *Learning rate* - Hyperparameter that determines the step size taken during each iteration of the mirror descent algorithm.
    * *Bregman* - Selects the corresponding mirror map for each of the four Bregman divergences used in this application.
        * When Mahalanobis is chosen, an additional input box for inputting the positive definite matrix Q is shown. Q should be inputted in the following format [a, b, c, d] for $\begin{bmatrix}a & b \\ c & d\end{bmatrix}$


---

#### *Figures*
The experiment tool automatically constructs 5 figures:
* The optimisation trajectory presented on a 2D contour plot as well as a 3D plot
* The optimisation trajectory in the dual space
* The gradient norm of parameters across iterations
* The bregman divergence between parameters (step size), this is measured in respect to the selected mirror map (e.g. if Euclidean is selected then this would represent the squared Euclidean distance).

Users can choose which figures to display by using the buttons above, allowing them to focus on specific figures. Users can also hover the cursor over either trajectory graph to view the corresponding values for that iteration from the other figures (this functionality is slightly laggy for the 3D simplex plots). Note that the 3D trajectory figure is automatically disabled for custom 1D functions.

---

#### *Metrics*
For each configuration, metrics are displayed on the right hand side of the page after an experiment has run. Note that any time Bregman Divergence is mentioned in a metric, it will be the Bregman used for that specific configuration. Users can click on a metric to highlight the corresponding row in each experiment table.
* **Step Sizes:** Measures the Euclidean distance between consecutive primal iterates.
* **Average Step Shrink Rate:** Measures the average ratio by which consecutive step sizes decrease.
* **Minimum Bregman Divergence:** Records the smallest Bregman divergence observed during the run.
* **Maximum Bregman Divergence:** Records the largest Bregman divergence observed during the run.
* **Mean Bregman Divergence:** Provides the average Bregman divergence over all iterations.
* **Average Bregman Shrink Rate:** Measures the overall average rate at which the Bregman divergence decreases.
* **Average Dual Step Shrink Rate:** Measures the average ratio by which dual space step sizes decrease.
* **Gradient Threshold Iteration:** Indicates the iteration when the gradient norm first falls below a set threshold (set to $0.001$) (This can sometimes be met without convergence especially in large flat valleys) (Displays as None if not met).
* **Gradient Log Slope:** Measures the rate of decrease of the gradient norm on a logarithmic scale (negative - the gradients are decaying exponentially which is usually an indicator of convergence / positive - gradients are increasing over iterations which may signal divergence or instabilitiy).
* **Minimum Gradient:** Records the smallest gradient norm observed during the run.
* **Maximum Gradient:** Records the largest gradient norm observed during the run.
* **Mean Gradient:** Provides the average gradient norm over all iterations.
* **Total Run Time:** Measures the total time taken to complete the experiment.
* **Average Iteration Time:** Measures the average time taken per iteration. (Displayed as None if not met)
* **Convergence Iteration:** Indicates the iteration number where the algorithm has converged to the optimum within a tolerance of $1 \times 10^{-6}$.  (Displayed as None if not met or using custom objective)
* **Distance to Optimum:** Measures the Euclidean distance from the final iterate to the known optimum. (Displayed as None if using custom objective)

*Following metrics are recorded as lists in the experiment.json file once the experiment has been downloaded*
* **Dual Step Sizes:** Measures the Euclidean distances between consecutive iterates in the dual space.
* **Bregman Shrink Rates:** Measures the per-iteration ratios of the Bregman divergence values.
* **Cosine Similarities:** Measures the cosine similarity between each step direction.

---

#### *Saving / Loading*
* Users can save experiments once the run has completed using the button at the bottom right of the Configuration panel. Note that as soon as any parameters in the configuration panel are changed, the save button is disabled until a run has been completed with the new configuration - this is to prevent saving with parameters not matching the currently displayed figures and metrics.
* Users can load any previously saved experiments using the button in the Loading panel of the experiment tool. 
* Users can also choose from 4 different pre-configured experiments designed to highlight where each mirror map is most effective. 

*** 
*Any questions, bugs or ammendments can be reported via email to **gabriel.downes@student.manchester.ac.uk** *

"""


layout = html.Div([
    html.H3("User Guide / Help"),

    dcc.Markdown(mirrormap_md, className="padding-markdown", mathjax=True)

], className="maps-desc")
