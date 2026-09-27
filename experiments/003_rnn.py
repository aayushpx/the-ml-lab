import torch
import torch.nn as nn

rnn = nn.RNN(
    input_size = 10,  # x_t \in R^10
    hidden_size = 32,  # h_t \in R^32
    num_layers = 1,  # one recurrent layer
    batch_first = True  # input shape: Batch, Timestep, Dimension
)

x = torch.randn(8, 5, 10)
# batch_size = 8
# sequence_length = 5
# hidden_size = 10

h0 = torch.zeros(1, 8, 32)
# num_layers = 1
# batch_size = 8
# hidden_size = 32

output, hn = rnn(x, h0)

print(output.shape)  # torch.Size([8, 5, 32])
print(hn.shape)  # torch.Size([1, 8, 32])
