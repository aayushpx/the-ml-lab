"""
Experiment 001: Learning XOR with a neural network.

A minimal PyTorch implementation of a two-layer neural network
trained with SGD to learn the XOR function.
"""

import torch
import torch.nn as nn
import torch.optim as optim # core

# XOR dataset
X = torch.tensor([ # X: inputs/features
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# targets/labels y, 2-dimensional array
y = torch.tensor([ 
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

print(X.shape)
print(y.shape)

# Defining network
class XORNet(nn.Module): # inherits from torch.nn.Module
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2) # Linear(input features, output features)
        self.output = nn.Linear(2, 1) # 2 input feature, 1 output feature
        self.sigmoid = nn.Sigmoid()   # sigmoid activation function

    def forward(self, x): # forward propagation of input x
        """Describe how input data moves through the network"""

        # self.hidden(x) performs the hidden layer's
        # weighted-sum calculation.
        hidden_net_input = self.hidden(x)

        # apply sigmoid separately to every hidden neuron value.
        h = self.sigmoid(hidden_net_input)

        # pass the two hidden outputs into the output layer. 
        output_net_input = self.output(h)

        # apply sigmoid to convert each output into a value 
        # 0 and 1
        z = self.sigmoid(output_net_input)

        return z

# fix random seed so initial parameter values are reproducible 
torch.manual_seed(0)

# create one XORNet object 
# calls XORNET.__init__()
model = XORNet()

print(model)


print("Hidden-layer weight shape:", model.hidden.weight.shape)
print("Hidden-layer bias shape:", model.hidden.bias.shape)

print("Output-layer weight shape:", model.output.weight.shape)
print("Output-layer bias shape:", model.output.bias.shape)

# create mean squared error loss function
criterion = nn.MSELoss()

# model.parameters() gives us the optimiser
optimizer = optim.SGD(model.parameters(), lr=0.5)

n_epochs = 10000


for epoch in range(n_epochs):
    optimizer.zero_grad()

    output = model(X)

    loss = criterion(output, y)

    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():

    predictions = model(X)

    binary_predictions = (predictions > 0.5).float()

    print("\nFinal raw outputs:")
    print(predictions)

    print("\nFinal binary predictions:")
    print(binary_predictions)

    print("\nCorrect targets:")
    print(y)
