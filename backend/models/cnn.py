import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    """
    Lightweight 3-layer Convolutional Neural Network (CNN) for Facial Emotion Recognition (FER).
    Inputs: Grayscale images of shape (Batch, 1, 48, 48)
    Outputs: Raw classification logits of shape (Batch, 7) representing the emotion categories.
    """
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 24 x 24
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 12 x 12
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 6 x 6
        
        # Classifier
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 6 * 6)
        
        # Classification
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model_size(model):
    """Calculates and returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Rapid shape validation and parameters count check
    model = EmotionCNN()
    test_input = torch.randn(2, 1, 48, 48)
    test_output = model(test_input)
    
    print("Model initialized successfully!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Trainable parameters: {get_model_size(model):,}")
