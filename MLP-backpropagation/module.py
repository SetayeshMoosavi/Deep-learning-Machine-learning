import numpy as np
from tensor import Tensor
from function import *

class Module:
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
               
        return params

    def set_parameters(self, params):
        param_list = self.parameters()
        for i, p_data in enumerate(params):
            param_list[i].data = p_data
    def forward(self, *args, **kwargs): ...
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
  
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        initial_weights = np.random.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
       
        self.weights = Tensor(initial_weights,True)

        initial_bias = np.zeros((1, self.out_features))
        self.bias = Tensor (initial_bias , True)

    def forward(self, x):
        return x @ self.weights + self.bias


class ReLU(Module):
  
    def forward(self, x):
         return ReLU.apply(x)