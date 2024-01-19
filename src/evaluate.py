import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MazeSolverCNN
from data_loader import MazeDataset

# Define your evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    print(f'Evaluation Results - Loss: {average_loss}, Accuracy: {accuracy}')

# Example usage
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # Add more transforms if needed
# ])
# test_dataset = MazeDataset(root_dir='path/to/your/project', transform=transform)
# test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# model = MazeSolverCNN()
# criterion = nn.CrossEntropyLoss()

# evaluate(model, test_data_loader, criterion)
