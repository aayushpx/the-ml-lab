import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size = 10,
    hidden_size = 32,
    num_layers = 1,
    batch_first = True
)

x = torch.randn(8, 5, 10)
h0 = torch.zeros(1, 8, 32)
c0 = torch.zeros(1, 8, 32)  # cell state that carries long term memory

output, (hn, cn) = lstm(x, (h0, c0))

print(output.shape)  # [8, 5, 32]
print(hn.shape)      # [1, 8, 32]  # final hidden state
print(cn.shape)      # [1, 8, 32]  # final cell state


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output, (hn, cn) = self.lstm(x)
            last_hidden = hn[-1]  # final representation that LSTM chose
            logits = self.fc(last_hidden)
            return logits
