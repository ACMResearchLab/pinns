import torchopt
import torch
from pinn_loss import loss_fn
from pinn_functional import f, params

# choose the configuration
batch_size = 30  # number of colocation points sampled in the domain
num_iter = 100  # maximum number of iterations
learning_rate = 1e-1  # learning rate
domain = (-5.0, 5.0)  # logistic equation domain

# choose optimizer with functional API using functorch
optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))
# train the model
for i in range(num_iter):

    # sample colocations points in the domain randomly at each epoch
    x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
    print("First x : ", x)

    # update the parameters using the functional API
    loss = loss_fn(x, params)
    params = optimizer.step(loss, params)

    print("Iteration {i} with loss {float(loss)}")