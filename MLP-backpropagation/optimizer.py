from tensor import Tensor
from typing import List
import numpy as np

class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")

        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:    
           if param.grad is not None:
                param.grad.fill(0)

class Momentum: 
    def __init__ (self ,params: List[Tensor], beta: float = 0.9 , lr: float = 0.01):
        self.params = params
        self.lr = lr
        self.v = [np.zeros_like(param.data) for param in self.params] 
        self.beta = beta

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.v[i]  = self.beta * self.v[i] - self.lr * param.grad 
                param.data += self.v[i] 
        
    def zero_grad(self):
        for param in self.params:    
           if param.grad is not None:
                param.grad.fill(0)


class Adam: 
    def __init__ (self,params: List[Tensor], lr: float = 0.01 , beta1: float = 0.9, beta2: float = 0.999 , epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = [np.zeros_like(param.data) for param in self.params] 
        self.m = [np.zeros_like(param.data) for param in self.params] 
        self.t = 0 

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.m[i]  = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad 
                self.v[i]  = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad**2) 

                # for bit corrected Adam we need to use these:
                self.m_hat = self.m[i] / (1 - self.beta1**self.t)
                self.v_hat = self.v[i] / (1 - self.beta2**self.t)

                param.data -= (self.lr * self.m_hat)/ (np.sqrt(self.v_hat) + self.epsilon)
    
    def zero_grad(self):
        for param in self.params:    
           if param.grad is not None:
                param.grad.fill(0)
