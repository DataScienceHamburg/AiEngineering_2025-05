#%% packages
import torch
import seaborn as sns
import numpy as np
#%% create tensor
x = torch.tensor(1.0, requires_grad=True)
x

#%% y = f(x)
y = (x-3) * (x-6) * (x-4)
y

#%% automatic gradients
y.backward()
# %%
x.grad


#%%
def y_function(val):
    return (val-3)*(val-6)*(val-4)

x_range = np.linspace(0, 10, 101)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y=y_range)
# %%

#%%
