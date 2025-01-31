import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_layers=True):
        """
        Initialize the ResNet50 model for vehicle classification.
        
        Args:
            num_classes (int): Number of output classes (e.g., car, bus, truck, etc.).
            freeze_layers (bool): Whether to freeze all layers except the final fully connected layer.
        """
        super(VehicleClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Freeze layers if required
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Adjust the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor (batch of images).
        
        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

# Example usage
if __name__ == "__main__":
    num_classes = 5  # car, bus, truck, motorcycle, bicycle
    
    model = VehicleClassifier(num_classes=num_classes, freeze_layers=True)
    
    print(model)
    
    # Example input tensor (batch of 3 RGB images of size 224x224)
    example_input = torch.randn(3, 3, 224, 224)
    
    # Forward pass
    output = model(example_input)
    print("Output shape:", output.shape)  # Should be [3, num_classes]