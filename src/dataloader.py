import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MazeDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def _len_(self):
        # Assuming all maze images are in the processed folder
        return len(os.listdir(os.path.join(self.root_dir, 'processed')))

    def _getitem_(self, idx):
        img_name = os.path.join(self.root_dir, 'processed', f'maze_{idx}.png')
        image = Image.open(img_name).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image

# Example usage of the data loader
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # Add more transforms if needed
# ])
# dataset = MazeDataset(root_dir='path/to/your/project', transform=transform)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
