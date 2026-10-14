import torch
import torch.nn as nn

# The external interface is very similar to RNN, but the 
# internal state update is gated
gru = nn.GRU(
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

output, hn = gru(x, h0)

print(output.shape)  # torch.Size([8, 5, 32])
print(hn.shape)  # torch.Size([1, 8, 32])


###### Using GRU for a Many-to-One Prediction Task

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # GRU extracts hidden representations
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = True
        )

        # Linear layer maps final hidden state to prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, D)
        output, hn = self.gru(x)  # hn stores the final hidden state of each GRU layer
        # for a many-to-one task, the final hidden state is used as a "summary"

        # output: (B, T, H)
        # hn: (L, B, H)

        last_hidden = hn[-1]  # slice final hidden state
        pred = self.fc(last_hidden)

        return pred
