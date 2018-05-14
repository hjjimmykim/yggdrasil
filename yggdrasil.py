import numpy as np
import matplotlib.pyplot as plt
import time

# Extract ohlcv data
from read_data_cmc import extract_data

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Parameters
torch.manual_seed(1) # Random seed
N_ep = 10 # Number of episodes
test_prop = 0.2 # Proportion of data set aside for test
alpha = 0.01 # Learning rate

# Model
class yggdrasil(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=100, num_layers=1):
        super(yggdrasil, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 1
        self.lstm1 = nn.LSTMCell(input_dim, hidden_dim)
        # LSTM 2
        self.lstm2 = nn.LSTMCell(hidden_dim, input_dim)
        
    def forward(self, ohlcv, future = 0):
        # ohlcv = num_timepoints x 5
        # outputs = (num_timepoints + future) x 5
    
        # Try to predict future steps after data end
        outputs = []
        
        # Initializations for lstm1
        h1_t = torch.zeros(self.input_dim, self.hidden_dim)
        c1_t = torch.zeros(self.input_dim, self.hidden_dim)
        # Initializations for lstm2
        h2_t = torch.zeros(self.input_dim, self.hidden_dim)
        c2_t = torch.zeros(self.input_dim, self.hidden_dim)
        
        # Iterate
        for i in range(ohlcv.size(0)):
            h1_t,c1_t = self.lstm1(ohlcv[i], (h1_t,c1_t))
            h2_t,c2_t = self.lstm2(h1_t, (h2_t,c2_t))
            outputs += [h2_t]
            
        # Predict the future
        for i in range(future):
            h1_t,c1_t = self.lstm1(ohlcv[i], (h1_t,c1_t))
            h2_t,c2_t = self.lstm2(h1_t, (h2_t,c2_t))
            outputs += [h2_t]
    
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs
        
# Extract data, define model and learning settings
o,h,l,c,v,cap = extract_data()
data = np.vstack([o,h,l,c,v])
data = data.T
data = torch.from_numpy(data).float() # Convert to tensor

num_data = len(data) # Number of data points
num_train = int((1-test_prop)*num_data) # Number of training data

# Train-test split
data_train = data[:num_train]
data_test = data[num_train:]

# Set up observations & targets
x_train = data_train[:-1] # Observations (won't use the last point to make a prediction)
y_train = data_train[1:]  # Targets (won't try to predict the first point)
x_test = data_test[:-1]
y_test = data_test[1:]

# Define model
model = yggdrasil(input_dim=5,hidden_dim=100,num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters(), lr = alpha)
        
# Training
t_start = time.time()
t_p1 = time.time()
for i_ep in range(N_ep):

    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Time stuff
    t_p2 = time.time()
    print('episode ' + str(i_ep) + ' time: ' + str(t_p2-t_p1))
    t_p1 = t_p2
    
t_finish = time.time()
print('Total time: ' + str(t_finish-t_start) + ' s')

train_output = model(x_train)

plt.figure()
ohlcv_train = y_train.T
ohlcv_train_pred = train_output.T
plt.plot(y_train[0],label='True data')
plt.plot(train_output[0],label='Predictions')

test_output = model(x_test)