import torch
import numpy as np 
from MirrorDescent import MirrorDescent
import plotly.graph_objects as plotly
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
torch.manual_seed(0)
from typing import Dict, Any
import time

class ExperimentMD():
    # results variable determines wheter the calculate_metrics function should calculate classification or regression metrics 
    # and whether this should be recorded/calculated or not 
    def __init__(self, objective = lambda x: x**2, bregman="EUCLID", results = 'R',
                gradient_calculation = "Classical",
                Q = torch.tensor([[10, 0.0], [0.0, 1.0]]),
                Q_inv=torch.tensor([[0.1,0.0], [0.0, 1.0]]),
                x_star=None, f_star = None, tolerance=1e-6, dim=2):
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
        # matrix for mahalanobis distance
        self.Q = torch.tensor(Q, dtype=torch.float64)
        self.Q_inv = torch.tensor(Q_inv, dtype=torch.float64) 
        self.dim = dim 
        self.dgfs = {
            'EUCLID' : lambda x: 0.5 * torch.sum(x**2),
            # domain x > 0 (probabilities) s.t. sum(x) = 1 (ideally)
            'KL' : lambda x: torch.sum((x+1e-8) * torch.log(x +1e-8) - (x + 1e-8)),
            'MAHALANOBIS': lambda x: 0.5 * torch.sum(x * (self.Q @ x)),
            'ITAKURA-SAITO': lambda x: -torch.sum(torch.log(x + 1e-8)),

        }
        # approximation logs
        self.metrics = []
        self.loss_logs = [] 
        self.prediction_data = {}

        # logs for approximation / minimisation
        self.gradient_logs = []
        self.avg_divergence_logs = []


        # minimisation logs
        self.minimisation_guesses = []
        # known minimiser and minimum
        self.x_star = x_star
        self.f_star = f_star
        # allowed tolerance for convergence 
        self.tolerance = tolerance
        self.objective_logs = []
        self.gap_logs = []
        self.dist_logs = []  
        self.iteration_times = []
        self.iteration_of_convergence = None
        self.iter_times = []
        self.total_run_time = None

    def clear(self):
        # function to clear any logs for the next experiment
        self.metrics = []
        self.loss_logs = [] 
        self.gradient_logs = []
        self.avg_divergence_logs = []
        self.minimisation_guesses = []
        self.prediction_data = {}
        self.objective_logs = []
        self.gap_logs = []
        self.dist_logs = []  
        self.iteration_times = []
        self.iteration_of_convergence = None
        self.iter_times = []
        self.total_run_time = None


    def gather_metrics(self, grad_threshold = 1e-3,step_ratio_threshold = 1e-15,
                       breg_ratio_threshold = 1e-15,dual_step_ratio_threshold = 1e-15):
        
        # ratio threshold params are to prevent division by 0 if the step size is extremely small

        metrics = {}
        """
        returns a dict in the following format
        {
        "step_sizes": []
        "cosine_similarities": [],
        "avg_step_shrink_rate": float or None,
        "bregman_shrink_rates": [],
        "avg_bregman_shrink_rate": float or None
        "dual_step_sizes": [],
        "avg_dual_step_shrink_rate": float or None,
        "grad_threshold_iteration": int or None,
        "gradient_log_slope": float or None,
        "min_grad": float or None,
        "max_grad": float or None
        "mean_grad": float or none,
        "min_bregman": float or None,
        "max_bregman": float or None,
        "mean_bregman": float or None,
        "total_run_time": float or none
        "average_iter_time": float or None,
        "convergence_iter": float or None,
        "distance_to_opt": float or None
        }
        """

        # PRIMAL SPACE STEP SIZES AND COSINE SIMILARITIES
        # this section calculates the euclidean step size, as well as the cosine similarity 
        # between each step
        step_sizes = []
        cosine_sims = []
        guesses = self.minimisation_guesses
        
        if len(guesses) < 2:
            metrics["step_sizes"] = []
            metrics["cosine_similarities"] = []
            metrics["avg_step_shrink_rate"] = None
        else:
            for t in range(1, len(guesses)):
                old_x = guesses[t-1]
                new_x = guesses[t]
                step_vec = new_x - old_x
                step_norm = np.linalg.norm(step_vec)
                step_sizes.append(step_norm)

                # cosine similarity with the previous step
                if t >= 2:
                    older_x = guesses[t-2]
                    prev_step_vec = old_x - older_x
                    dot_ = np.dot(step_vec, prev_step_vec)
                    norm_ = (np.linalg.norm(step_vec) * np.linalg.norm(prev_step_vec))
                    cosine_sims.append(dot_/norm_ if norm_ > 1e-15 else 0.0)
                else:
                    cosine_sims.append(None)

            # step size shrink rate
            step_ratios = []
            for i in range(1, len(step_sizes)):
                if step_sizes[i-1] > step_ratio_threshold:
                    ratio = step_sizes[i]/step_sizes[i-1]
                    step_ratios.append(ratio)
            avg_step_shrink_rate = np.mean(step_ratios) if len(step_ratios) > 0 else None

            metrics["step_sizes"] = step_sizes
            metrics["cosine_similarities"] = cosine_sims
            metrics["avg_step_shrink_rate"] = avg_step_shrink_rate

        # this section calculates statistics for each steps in terms of the bregman divergence
        # corresponding to the mirror map, calculates min, max, mean, and shrink rate
        div_logs = self.avg_divergence_logs
        if len(div_logs) == 0:
            metrics["bregman_shrink_rates"] = []
            metrics["avg_bregman_shrink_rate"] = None
            metrics["min_bregman"] = None
            metrics["max_bregman"] = None
            metrics["mean_bregman"] = None
        else:
            breg_array = np.array(div_logs, dtype=float)
            # basic stats
            metrics["min_bregman"] = float(breg_array.min())
            metrics["max_bregman"] = float(breg_array.max())
            metrics["mean_bregman"] = float(breg_array.mean())

            # bregman shrink rate
            breg_shrink_rates = []
            for i in range(1, len(breg_array)):
                prev_div = breg_array[i-1]
                if prev_div > breg_ratio_threshold:
                    ratio = breg_array[i]/prev_div
                    breg_shrink_rates.append(ratio)
            avg_breg_shrink_rate = np.mean(breg_shrink_rates) if len(breg_shrink_rates) > 0 else None
            
            metrics["bregman_shrink_rates"] = breg_shrink_rates
            metrics["avg_bregman_shrink_rate"] = avg_breg_shrink_rate

        # this section calculates statistics for steps in the dual space,
        # just uses the euclidean norm rather than the specific bregmans, as these values should already 
        # exist in the bregman-specific geometry from the different mirror maps 
        dual_logs = None
        if hasattr(self, 'optimiser') and hasattr(self.optimiser, 'logs'):
            dual_logs = self.optimiser.logs.get('dual', None)
        
        if not dual_logs or len(dual_logs) < 2:
            # incase dual logs aren't present, i.e. logs=False when initialising MirrorDescent class
            metrics["dual_step_sizes"] = []
            metrics["avg_dual_step_shrink_rate"] = None
        else:
            dual_step_sizes = []
            for t in range(1, len(dual_logs)):
                old_y = np.array(dual_logs[t-1], dtype=float)
                new_y = np.array(dual_logs[t], dtype=float)
                delta_y = new_y - old_y
                dual_step_sizes.append(float(np.linalg.norm(delta_y)))

            # dual space step size shrink rate
            dual_step_ratios = []
            for i in range(1, len(dual_step_sizes)):
                if dual_step_sizes[i-1] > dual_step_ratio_threshold:
                    ratio = dual_step_sizes[i]/dual_step_sizes[i-1]
                    dual_step_ratios.append(ratio)

            avg_dual_step_shrink_rate = np.mean(dual_step_ratios) if len(dual_step_ratios) > 0 else None
            metrics["dual_step_sizes"] = dual_step_sizes
            metrics["avg_dual_step_shrink_rate"] = avg_dual_step_shrink_rate

        # this section calculates gradient norm metrics: min, max, mean,
        # gradient threshold iteration - the iteration at which the gradient norm reaches a certain threshold to indicate
        # stationarity, but this could be misleading in non-convex functions/ functions with multiple local minima or flat regions
        # also calculates the gradient log slope which approximates how quickly the gradients decrease on a log scale
        # as a function of the iteration number
        grad_logs = np.array(self.gradient_logs, dtype=float) if len(self.gradient_logs) > 0 else None
        if grad_logs is not None and grad_logs.size > 0:
            # gradient threshold iteration
            grad_threshold_iteration = None
            for i, gnorm in enumerate(grad_logs):
                if gnorm < grad_threshold:
                    grad_threshold_iteration = i
                    break
            metrics["grad_threshold_iteration"] = grad_threshold_iteration

            # log slope
            safe_grad = grad_logs.copy()
            safe_grad[safe_grad <= 0] = 1e-15
            log_grad = np.log(safe_grad)
            xvals = np.arange(len(log_grad))
            if len(log_grad) > 1:
                mean_x = xvals.mean()
                mean_y = log_grad.mean()
                num = np.sum((xvals - mean_x) * (log_grad - mean_y))
                den = np.sum((xvals - mean_x)**2)
                slope = num/den if den != 0 else None
                metrics["gradient_log_slope"] = float(slope) if slope is not None else None
            else:
                metrics["gradient_log_slope"] = None

            # min, max, mean
            metrics["min_grad"] = float(grad_logs.min())
            metrics["max_grad"] = float(grad_logs.max())
            metrics["mean_grad"] = float(grad_logs.mean())
        else:
            metrics["grad_threshold_iteration"] = None
            metrics["gradient_log_slope"] = None
            metrics["min_grad"] = None
            metrics["max_grad"] = None
            metrics["mean_grad"] = None

        # this section calculates run times
        metrics["total_run_time"] = self.total_run_time
        metrics["average_iter_time"] = np.mean(self.iter_times)

        # finally if the optimum is known, calculate distance from final step to optimum
        # this just uses the euclidean distance as its a performance measure to be compared with other experiments
        # convergence rate - how many iterations to converge
        if self.x_star is not None and self.f_star is not None:
            metrics["convergence_iter"] = self.iteration_of_convergence
            metrics["distance_to_opt"] = np.linalg.norm(self.minimisation_guesses[-1] - np.array(self.x_star))

        return metrics

    
    def construct_optimiser(self, params, lr):
        # function to construct the optimiser from the mirror descent class
        print(self.Q)
        self.optimiser = MirrorDescent(params, lr, self.bregman, self.Q, self.Q_inv, logs=True)


    def run_experiment_minimise(self, init, iter, lr):
        # minimises an objective function using the MirrorDescent optimiser
        # first convert the starting point to a tensor
        self.minimisation_guesses.append(init)
        simplex = False
        if isinstance(init, list):
            if self.dim == 3: simplex = True

        
        x = torch.tensor(init, requires_grad=True, dtype=torch.float64)

        self.construct_optimiser([x], lr)
       
        overall_start_time = time.time()

        print("Optimiser constructed, now starting minimisation")
        for i in range(iter):
            self.optimiser.zero_grad()
            iter_start_time = time.time()
            
            
            # for the 1d case where x is a 0d tensor and cannot iterate over with *x
            if isinstance(init, list):
                y = self.objective(*x)
            else:
                y = self.objective(x)

            y.backward()

            # recording gradient norms
            self.calculate_record_gradient_norms()
            
            # logging objective value
            f_val = y.item()
            self.objective_logs.append(f_val)
            
            # record the gap between current objective value and minimum objective value (if known)
            if self.f_star is not None:
                
                gap = f_val - self.f_star
                self.gap_logs.append(gap)
            else:
                gap = None

            # record current params before stepping for bregman divergence calculatation
            old_params = []
            for group in self.optimiser.param_groups:
                for param in group['params']:
                    old_params.append(param.data.clone())

            self.optimiser.step()


            # normalises the probabilties after step is performed if the obejctive is the simplex objective
            if simplex:
                x.data = x.data / x.data.sum()
            #recording the iteration run time
            iter_elapsed = time.time() - iter_start_time
            self.iter_times.append(iter_elapsed)

            # check if we are within the tolerance threshold of the minimum, record this
            # as the iteration of convergence if so 
            if gap is not None and abs(gap) < self.tolerance:
                if self.iteration_of_convergence is None:
                    self.iteration_of_convergence = i

                    print(f"Tolerance reached at iteration {i}, terminating run")
                    self.calculate_record_average_bregman_divergence(old_params, self.bregman)
                    self.minimisation_guesses.append(x.detach().cpu().numpy().copy())
                    break
            
            # calculate and record the bregman divergence between params of this iteration and the next
            self.calculate_record_average_bregman_divergence(old_params, self.bregman)

            self.minimisation_guesses.append(x.detach().cpu().numpy().copy())
        self.total_run_time = time.time() - overall_start_time
        print(f"Iterations for minimisation complete. Total time: {self.total_run_time:.4f} s")

    
    def calculate_record_gradient_norms(self):
        # function calculating the euclidean norm of the gradients
        grad_norm = 0.0
        for group in self.optimiser.param_groups:
            for param in group['params']:
                if param.grad is not None: 
                    grad_norm += param.grad.data.norm().item()**2
        grad_norm = grad_norm**0.5
        self.gradient_logs.append(grad_norm)

    def calculate_record_average_bregman_divergence(self, old_params, bregman):
        # function for calculating the bregman divergence between parameters of the current
        # epoch/iteration, and the parameters after optimiser.step() is
        total_divergence = 0.0
        params = 0 
        idx = 0 
        psi = self.dgfs[bregman]
        # loop through each parameter recording the divergence across a step for each
        # for approximation, results in the average divergence accross each parameter
        # for minimisation, there is only one "parameter" - the [x]/ [x,y] / [p1,p2,p3] so just the pure b divergence between two points 
        for group in self.optimiser.param_groups:
            for param in group['params']:
                # psi(x) - psi(y) - <psi'(y) , x - y>
                previous = old_params[idx]
                psi_y = psi(previous)                   
                psi_x = psi(param)

                # inner product
                product = torch.sum(self.optimiser.grad_psi(previous) * (param-previous))

                # calculating the bregman divergence
                divergence = psi_x - psi_y - product 
                total_divergence += divergence

                params += 1
                idx +=1 
        avg_divergence = total_divergence / params 
        avg_divergence = avg_divergence.cpu().detach().numpy()
        self.avg_divergence_logs.append(avg_divergence)







    
    
    
            



