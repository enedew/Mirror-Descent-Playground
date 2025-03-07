import torch

# A number of preset functions which the user can customise, each with inherent disadvantages/advantages 
# for each mirror map 


# anisotropic objective function - designed to work best with mahalanobis distance
# if positive definite matrix is set to [[a, 0][0, b]] the optimisation trajectory will be a straight line
# whereas euclidean will follow a curved trajectory.
def anisotropic_quadratic(x, a=10.0, b=1.0, optimum=torch.tensor([2.0, 3.0])):
    return a * (x[0] - optimum[0])**2 + b * (x[1] - optimum[1])**2

# objective on the simplex, just KL divergence of weights to target distribution
def simplex_objective(p, weights=torch.tensor([0.2, 0.3, 0.5])):
    eps = 1e-8
    return torch.sum(weights * torch.log((weights) / (p + eps)))


# Rosenbrock function - classic nonconvex function for testing performance of optimisation algorithms 
def rosenbrock(x, a=1.0, b=1.0):
    return (a - x[0])**2 + b*(x[1] -x[0]**2)**2


# quartic function - 2d convex function of a higher order
def log_sum_exp(x):
    return torch.log(torch.exp(x[0]) + torch.exp(x[1]))