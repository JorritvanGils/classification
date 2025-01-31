import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from model import VehicleClassifier  # Import the model we created earlier

# Custom Dataset Class
class VehicleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for vehicle images.
        
        Args:
            data_dir (str): Path to the directory containing the processed images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load image paths and labels
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.labels.append(int(label))  # Assuming labels are integers
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training Function
def train_model(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use (e.g., "cuda" or "cpu").
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# Validation Function
def validate_model(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use (e.g., "cuda" or "cpu").
    
    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# Main Script
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_classes = 5  # Number of vehicle classes
    data_dir = "/media/jorrit/ssd/Classification/data/processed"  # Path to processed data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load Dataset
    train_dataset = VehicleDataset(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    model = VehicleClassifier(num_classes=num_classes, freeze_layers=True).to(device)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    
    # Save the Trained Model
    torch.save(model.state_dict(), "vehicle_classifier.pth")
    print("Model saved to vehicle_classifier.pth")