#%% packages
import torch
import seaborn as sns
import numpy as np

#%% tensor creation
x = torch.tensor(5.5)
# %% simple graph y=f(x)
y = x + 10

# %%
x.requires_grad  # gradient calculation is deactivated by default

#%% activate gradient 
x.requires_grad_()
x.requires_grad
# %% 2nd example
x = torch.tensor(1.0, requires_grad=True)
y = (x - 3) * (x- 6) * (x-4)
y.backward()
# %%
x.grad