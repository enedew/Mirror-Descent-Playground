import torch
from torch.optim import Optimizer 

# dictionary of mirror maps for different bregman divergences 
mirror_map_dict = { 
    # domain R^n
    'EUCLID' : lambda x: x,
    # domain x > 0 (probabilities) s.t. sum(x) = 1 (ideally)
    'KL' : lambda x: torch.log(x + 1e-8),
}

# dictionary of inverse mirror maps for different bregman divergences 

inv_mirror_map_dict = {
    'EUCLID' : lambda x: x,
    'KL' : lambda x: torch.exp(x + 1e-8) 
}


class MirrorDescent(Optimizer):
    def __init__(self, params, lr=0.01, bregman='EUCLID'):
        # set the mirror map and its inverse to the corresponding functions 
        self.grad_psi = mirror_map_dict[bregman]
        self.inv_grad_psi = inv_mirror_map_dict[bregman]

        # create a dict of hyperparameters
        defaults = dict(lr=lr)

        # merges model parameters and hyperparameters into a self.param_groups
        # this is a list containing the groups of model parameters and their corresponding
        # hyperparameters, e.g. if i need different hyperparameters for different layers in a model?
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            # loop through each parameter
            for param in group['params']:
                if param.grad is None: 
                    continue
                
                # get the gradient of the parameter 
                grad = param.grad.data

                # compute mirror map of param
                y0 = self.grad_psi(param.data) 

                #update in dual space 
                y1 = y0 - (lr * grad)

                # map back to primal space and update parameter value 
                param.data = self.inv_grad_psi(y1) 

                 





