import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_c=3, output=5, dropout=0.5):
        super(SimpleCNN, self).__init__()

        
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )

        
        self._calculate_fc_input_size(input_size=100)  

        # Fully connected layer
        self.fc = nn.Linear(self.flattened_size, output)
        self.softmax = nn.Softmax(dim=1)

    def _calculate_fc_input_size(self, input_size):

        """Calculate the input size of the fully connected layer dynamically."""

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)  # Simulasi input
            x = self.convBlock1(dummy_input)
            x = self.convBlock2(x)
            self.flattened_size = x.view(-1).shape[0]  # Flattened size

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(torch.flatten(x, 1))  # Flatten along batch dimension
        x = self.softmax(x)
        return x