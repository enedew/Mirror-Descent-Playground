import torch
from abc import ABC, abstractmethod
# A number of preset functions which the user can customise, each with inherent disadvantages/advantages 
# for each mirror map 


def differentiable_noise(x, x_opt, noise_std, frequency=5):
    # adds a cosine perturbation which fades towards the optimum
    x_opt = x_opt.view(-1, *([1]*(x.dim()-1)))
    diff = x - x_opt
    noise = noise_std * (1 - torch.cos(frequency * diff)).sum(dim=0)
    return noise

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
            noise = differentiable_noise(x_tensor, self.optimum, self.noise_std)
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

# rosenbrock function - classic nonconvex function for testing performance of optimisation algorithms 
class Rosenbrock(ObjectiveFunction):
    def __init__(self, a=1.0, b=100.0, noise_std=0.0):
        # the global minimum is (a, a**2) 
        super().__init__(optimum=torch.tensor([a, a**2]), noise_std=noise_std,name="Rosenbrock")
        self.a = a
        self.b = b

    def _compute(self, x):
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2


# rastrigin function - nonconvex with many local minima
class Rastrigin(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        super().__init__(optimum=torch.tensor([0.0, 0.0]), noise_std=noise_std, name="Rastrigin")
    
    def _compute(self, x):
        return 20 + (x[0]**2 - 10*torch.cos(2*torch.pi*x[0])) + (x[1]**2 - 10*torch.cos(2*torch.pi*x[1]))

# booth function - large flat narrow valley where the optimum lies
class Booth(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        # booth function has its optimum at (1, 3)
        super().__init__(optimum=torch.tensor([1.0, 3.0]),noise_std=noise_std, name="Booth")

    def _compute(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
# another function for optimisation benchmarking - gotten from wikipedia
class Ackley(ObjectiveFunction):
    def __init__(self, noise_std=0.0):
        # ackley function has optimum at 0,0
        super().__init__(optimum=torch.tensor([0.0,0.0]), noise_std=noise_std, name="Ackley")
    
    def _compute(self, x):
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * ( x[0]**2 + x[1]**2 )))
        term2 = -torch.exp(0.5 * (torch.cos(2*torch.pi*x[0]) + torch.cos(2*torch.pi*x[1])))
        return term1 + term2 + 20 + torch.exp(torch.tensor(1.0, dtype=x.dtype, device=x.device))

# cubic objective function
class CubicObjective(ObjectiveFunction):
    def __init__(self, optimum=torch.tensor([1.0, 1.0]), noise_std=0.0):
        super().__init__(optimum=optimum, name="2D Cubic Objective", noise_std=noise_std)
    
    def _compute(self, x):
        
        term1 = torch.pow(torch.abs(x[0] - self.optimum[0]), 3)
        term2 = torch.pow(torch.abs(x[1] - self.optimum[1]), 3)
        return (1/3) * (term1 + term2)

# exponential objective 
class ExponentialObjective2D(ObjectiveFunction):
    def __init__(self, optimum=torch.tensor([1.0, 1.0]), noise_std=0.0):
        super().__init__(optimum=optimum, name="2D Exponential Objective", noise_std=noise_std)
    
    def _compute(self, x):
        # f(x) = exp(x[0]-opt[0]) - (x[0]-opt[0]) + exp(x[1]-opt[1]) - (x[1]-opt[1])
        term1 = torch.exp(x[0] - self.optimum[0]) - (x[0] - self.optimum[0])
        term2 = torch.exp(x[1] - self.optimum[1]) - (x[1] - self.optimum[1])
        return term1 + term2

# function suited for itakura mirror map, basically just the divergence   
class ItakuraObjective(ObjectiveFunction):
    def __init__(self, a=1.0, lam=1.0, noise_std=0.0):
        # The optimum now will be at x=y=sqrt(a)
        optimum = torch.tensor([a**0.5, a**0.5], dtype=torch.float64)
        super().__init__(optimum=optimum, noise_std=noise_std, name="Modified Itakura Objective")
        self.a = a
        self.lam = lam
        self.eps = 1e-8  # avoid division by 0

    def _compute(self, x):
        # x should be a tensor of shape [2], where x[0] = x and x[1] = y.
        product = x[0] * x[1] + self.eps
        is_part = self.a / product - torch.log(self.a / product) - 1
        reg_part = self.lam * (x[0] - x[1])**2
        return is_part + reg_part