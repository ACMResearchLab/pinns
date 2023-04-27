from functorch import make_functional, grad, vmap
import torch
from pinn_nn_pytorch import NNApproximator

# create the PINN model and make it functional using functorch utilities
model = NNApproximator()
fmodel, params = make_functional(model)
print("Fmodel : ", fmodel)
print("Params : ", params)

def f(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    # only a single element is supported thus unsqueeze must be applied
    # for batching multiple inputs, `vmap` must be used as below
    x_ = x.unsqueeze(0)
    print("X underscore is ",x_)
    res = fmodel(params, x_).squeeze(0)
    print("\n\n\nIT WORKED!!!!\n\n\n")
    return res

# use `vmap` primitive to allow efficient batching of the input
f_vmap = vmap(f, in_dims=(0, None))

# return function for computing higher order gradients with respect
# to input by simply composing `grad` calls and use again `vmap` for
# efficient batching of the input
dfdx = vmap(grad(f), in_dims=(0, None))
d2fdx2 = vmap(grad(grad(f)), in_dims=(0, None))