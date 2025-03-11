import torch
from abc import ABC, abstractmethod
# A number of preset functions which the user can customise, each with inherent disadvantages/advantages 
# for each mirror map 




# class for objective functions
class ObjectiveFunction(ABC):
    def __init__(self, optimum, name="", noise_std=0.0):
        self.optimum = optimum
        self.name = name
        self.noise_std = noise_std


    @abstractmethod
    def _compute(self, x):
        """Compute the deterministic part of the function. 
           x is expected to be a torch tensor.
        """
        pass
    
    def __call__(self, *x):
        # if a single tensor is passed, use it directly 
        if len(x) == 1 and isinstance(x[0], torch.Tensor):
            x_tensor = x[0]
        # if all arguments are tensors then stack them 
        elif all(isinstance(arg, torch.Tensor) for arg in x):
            x_tensor = torch.stack(x)
        else:
            # else create new tensor from the inputs
            x_tensor = torch.tensor(x, dtype=torch.float64)
        value = self._compute(x_tensor)
        if self.noise_std > 0:
            x_tensor = x_tensor.to(torch.float64)
            noise = torch.randn(1, dtype=x_tensor.dtype, device=x_tensor.device) * self.noise_std
            value = value + noise
        return value

# anisotropic objective function - designed to work best with mahalanobis distance
# if positive definite matrix is set to [[a, 0][0, b]] the optimisation trajectory will be a straight line
# whereas euclidean will follow a curved trajectory.
class AnisotropicQuadratic(ObjectiveFunction):
    def __init__(self, a=10.0, b=1.0, optimum=torch.tensor([2.0, 3.0]), noise_std=0.0):
        super().__init__(optimum=optimum, noise_std=noise_std,name="Anisotropic Quadratic")
        self.a = a
        self.b = b

    def _compute(self, x):
        return self.a * (x[0] - self.optimum[0])**2 + self.b * (x[1] - self.optimum[1])**2

# objective on the simplex, just KL divergence of weights to target distribution
class SimplexObjective(ObjectiveFunction):
    def __init__(self, weights=torch.tensor([0.2, 0.3, 0.5]), noise_std=0.0):
        # the optimum is reached when p equals the target distribution (weights)
        super().__init__(optimum=weights, noise_std=noise_std,name="Simplex Objective")
        self.weights = weights
        self.eps = 1e-8

    def _compute(self, p):
        return torch.sum(self.weights * torch.log(self.weights / (p + self.eps)))

# Rosenbrock function - classic nonconvex function for testing performance of optimisation algorithms 
class Rosenbrock(ObjectiveFunction):
    def __init__(self, a=1.0, b=100.0, noise_std=0.0):
        # the global minimum is (a, a**2) 
        super().__init__(optimum=torch.tensor([a, a**2]), noise_std=noise_std,name="Rosenbrock")
        self.a = a
        self.b = b

    def _compute(self, x):
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2


# Rastrigin function - nonconvex with many local minima
class Rastrigin(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        super().__init__(optimum=torch.tensor([0.0, 0.0]), noise_std=noise_std, name="Rastrigin")
    
    def _compute(self, x):
        return 20 + (x[0]**2 - 10*torch.cos(2*torch.pi*x[0])) + (x[1]**2 - 10*torch.cos(2*torch.pi*x[1]))

class Booth(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        # booth function has its optimum at (1, 3)
        super().__init__(optimum=torch.tensor([1.0, 3.0]),noise_std=noise_std, name="Booth")

    def _compute(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    

class Ackley(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        # ackley function has optimum at 0,0
        super().__init__(optimum=torch.tensor([0.0,0.0]), noise_std=noise_std, name="Ackley")
    
    def _compute(self, x):
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * ( x[0]**2 + x[1]**2 )))
        term2 = -torch.exp(0.5 * (torch.cos(2*torch.pi*x[0]) + torch.cos(2*torch.pi*x[1])))
        return term1 + term2 + 20 + torch.exp(torch.tensor(1.0, dtype=x.dtype, device=x.device))
