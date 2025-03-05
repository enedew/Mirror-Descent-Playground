import torch
from torch.optim import Optimizer 

# defining some matrix Q for mahalanobis distance 
Q = torch.tensor([[2.0, 0.0],
                  [0.0, 1.0]])
Q_inv = torch.tensor([[0.5, 0.0],
                      [0.0, 1.0]])

# dictionary of mirror maps for different bregman divergences 
mirror_map_dict = { 
    # domain R^n
    'EUCLID' : lambda x: x,
    'MAHALANOBIS': lambda x: torch.matmul(Q, x),
    # domain x > 0 (probabilities) s.t. sum(x) = 1 (ideally)
    'KL' : lambda x: torch.log(x + 1e-8)+1,
    'ITAKURA-SAITO': lambda x: -1.0/x
} 

# dictionary of inverse mirror maps for different bregman divergences 

inv_mirror_map_dict = {
    'EUCLID' : lambda x: x,
    'MAHALANOBIS': lambda x: torch.matmul(Q_inv, x),
    'KL' : lambda x: torch.exp(x-1),
    'ITAKURA-SAITO': lambda x: -1.0/x 
}


class MirrorDescent(Optimizer):
    def __init__(self, params, lr=0.01, bregman='EUCLID', logs=True):
        # set the mirror map and its inverse to the corresponding functions 
        self.grad_psi = mirror_map_dict[bregman]
        self.inv_grad_psi = inv_mirror_map_dict[bregman]

        # create a dict of hyperparameters
        defaults = dict(lr=lr)
        self.logs = {'primal': [], 'dual': []} if logs else None
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
                
                # log values if required
                if self.logs is not None:
                    primal_val = param.data.clone().detach()
                    dual_val = y1
                    self.logs['primal'].append(primal_val.numpy())
                    self.logs['dual'].append(dual_val.numpy())
                    

                 





