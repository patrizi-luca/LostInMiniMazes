import torch
import torch.nn as nn
import torch.optim as optim

class MazeSolverCNN(nn.Module):
    def _init_(self):
        super(MazeSolverCNN, self)._init_()
        # Define your CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, num_classes)  # num_classes: Number of possible actions

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 11 * 11)  # Adjust the size based on your CNN architecture
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage of the model
# model = MazeSolverCNN()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
