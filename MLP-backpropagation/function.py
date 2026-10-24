import numpy as np

__all__ = ["Add", "Mul", "Pow", "Log", "Sum", "MatMul", "ReLU", "Sigmoid", "Softmax", "CrossEntropyWithSoftmax"]

class Context:
    def __init__(self, grad_fn, *inputs):
        self.grad_fn = grad_fn
        self.inputs = inputs

            
class Function:
    @staticmethod
    def forward(ctx, *args): ...
    
    @staticmethod
    def backward(ctx, upstream_grad): ...

    @classmethod
    def apply(cls, *args):
        from tensor import Tensor
        inputs = [a for a in args if isinstance(a, Tensor)]
        ctx = Context(cls, *inputs)
        ctx.needs_input_grad = tuple(t.requires_grad for t in inputs)
        requires_grad = any(ctx.needs_input_grad)

        output_data = cls.forward(ctx, *args)

        ctx = ctx if requires_grad else None
        
        output_tensor = Tensor(output_data, requires_grad, ctx)
        
        return output_tensor

def unbroadcast_to(a, shape):
    while len(a.shape) > len(shape):
        a = a.sum(axis=0)
    
    for i, dim in enumerate(shape):
        if dim == 1:
            a = a.sum(axis=i, keepdims=True)

    return a

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data + b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
    
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data * b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(ctx.b.data * upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(ctx.a.data * upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data @ b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad @ ctx.b.data.T if ctx.needs_input_grad[0] else None
        grad_b = ctx.a.data.T @ upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
            
class Pow(Function):
    @staticmethod
    def forward(ctx, base, exp):
        ctx.base = base
        ctx.exp = exp
        ctx.output = np.pow(base.data, exp.data)
        return ctx.output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_base = ctx.exp.data * np.pow(ctx.base.data, ctx.exp.data - 1) * upstream_grad if ctx.needs_input_grad[0] else None
        grad_exp = np.log(ctx.base.data) * ctx.output * upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_base, grad_exp)

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        return np.log(a.data)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = 1/ctx.a.data * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a, None)
        
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.a = a
        return np.sum(a.data, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = np.broadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        return (grad_a, None)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        ctx.mask = (a.data > 0)
        return np.maximum(a.data,0)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ctx.mask * upstream_grad if ctx.needs_input_grad[0] else None     
        return (grad_a, None)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        ctx.output = 1 / (1 + np.exp(-a.data))
        return ctx.output

    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ctx.output *(1 - ctx.output) * upstream_grad if ctx.needs_input_grad[0] else None        
        return (grad_a, None)

class Softmax(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.a = a
        ctx.axis = axis
        a_stable = a.data - a.data.max(axis=axis, keepdims=True)
        exp = np.exp(a_stable)
        ctx.output = exp / np.sum(exp, axis=axis, keepdims=True)     
        return ctx.output        
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ctx.output * (upstream_grad - np.sum(upstream_grad * ctx.output, axis=ctx.axis, keepdims=True)) if ctx.needs_input_grad[0] else None        
        return (grad_a, None)

class CrossEntropyWithSoftmax(Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        logits = y_pred.data
        ctx.y_true = y_true.data

        batch_size = logits.shape[0]
        ctx.batch_size = batch_size
        
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_stable)
        ctx.softmax_output = exp / np.sum(exp, axis=1, keepdims=True)     

        cross_entropy_loss = -np.sum(y_true.data * np.log(ctx.softmax_output + 1e-12))
        
        Normalized_Loss = cross_entropy_loss / batch_size
        return np.array(Normalized_Loss, dtype=np.float32)
       

    @staticmethod
    def backward(ctx, upstream_grad):
        grad = (ctx.softmax_output - ctx.y_true) / ctx.batch_size
        grad_a = grad * upstream_grad if ctx.needs_input_grad[0] else None        
        return (grad_a, None)