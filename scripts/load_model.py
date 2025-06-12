#%% packages
import torch

#%%
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x

# %%
model = LinearRegression(input_size=30, output_size=1, hidden_size=50)
# %% check the model weights
model.state_dict()
# %% load trained model weights
model.load_state_dict(torch.load('model1_dict.pt'))

# %%
