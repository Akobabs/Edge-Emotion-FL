import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define standard emotion labels in FER-2013
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Mapping Kaggle subfolder names to standard FER-2013 class indices
CLASS_TO_IDX = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

class FER2013Dataset(Dataset):
    """Custom PyTorch Dataset for FER-2013 (CSV/Synthetic fallback mode)."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure correct shape: (1, 48, 48) and normalized to [0, 1]
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dim
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def generate_synthetic_fer(num_samples=10000):
    """Generates synthetic FER-2013 data of 48x48 grayscale images for fallback."""
    print(f"Generating {num_samples} synthetic 48x48 facial emotion images for simulation fallback...")
    
    # Generate random pixel data (0 to 255)
    images = np.random.randint(0, 256, size=(num_samples, 48, 48), dtype=np.uint8) / 255.0
    
    # Generate labels (0 to 6) with a slightly unbalanced distribution to mimic real emotions
    probabilities = [0.13, 0.02, 0.14, 0.25, 0.17, 0.11, 0.18]
    labels = np.random.choice(7, size=num_samples, p=probabilities)
    
    return images, labels

def load_fer2013(csv_path=None):
    """Loads raw FER-2013 data from CSV or falls back to synthetic data."""
    if csv_path and os.path.exists(csv_path):
        try:
            print(f"Loading dataset from CSV: {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Extract labels and pixels
            labels = df['emotion'].values
            pixel_strings = df['pixels'].values
            
            # Parse pixels
            images = []
            for pix_str in pixel_strings:
                img = np.fromstring(pix_str, dtype=np.uint8, sep=' ').reshape(48, 48) / 255.0
                images.append(img)
            images = np.array(images, dtype=np.float32)
            
            # Partition based on 'Usage' if available
            if 'Usage' in df.columns:
                train_mask = df['Usage'] == 'Training'
                test_mask = df['Usage'].isin(['PublicTest', 'PrivateTest'])
                
                train_imgs, train_lbls = images[train_mask], labels[train_mask]
                test_imgs, test_lbls = images[test_mask], labels[test_mask]
                return train_imgs, train_lbls, test_imgs, test_lbls
            else:
                split_idx = int(0.8 * len(images))
                return images[:split_idx], labels[:split_idx], images[split_idx:], labels[split_idx:]
        except Exception as e:
            print(f"Failed to parse CSV due to error: {e}. Falling back to synthetic data.")
    
    # Fallback to synthetic
    train_imgs, train_lbls = generate_synthetic_fer(num_samples=8000)
    test_imgs, test_lbls = generate_synthetic_fer(num_samples=2000)
    return train_imgs, train_lbls, test_imgs, test_lbls

def partition_non_iid_dirichlet(labels, num_clients, alpha=0.5):
    """
    Partitions the dataset across clients using a Dirichlet distribution for label skew.
    """
    num_classes = 7
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples across clients using Dirichlet distribution
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        # Draw proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = np.array([p * len(idx_c) for p in proportions])
        proportions = np.round(proportions).astype(int)
        
        # Correct rounding differences
        diff = len(idx_c) - sum(proportions)
        if diff != 0:
            proportions[np.argmax(proportions)] += diff
            
        # Distribute indices
        start = 0
        for k in range(num_clients):
            end = start + proportions[k]
            client_indices[k].extend(idx_c[start:end])
            start = end
            
    # Shuffle indices for each client to mix classes
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
        
    return client_indices

def get_federated_loaders(csv_path=None, num_clients=5, alpha=0.5, batch_size=32):
    """
    Prepares Non-IID federated PyTorch DataLoaders.
    Checks for the Kaggle image directory first, then CSV, then falls back to synthetic data.
    """
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    # Check if raw Kaggle folders exist
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"[DATA] Found Kaggle FER-2013 image directories at {train_dir} and {test_dir}!")
        
        # ImageFolder transformation (convert to single channel, resize to 48x48, normalize [0, 1])
        img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
        
        try:
            train_dataset = ImageFolder(root=train_dir, transform=img_transform, class_to_idx=CLASS_TO_IDX)
            test_dataset = ImageFolder(root=test_dir, transform=img_transform, class_to_idx=CLASS_TO_IDX)
            
            print(f"[DATA] Successfully loaded {len(train_dataset)} training and {len(test_dataset)} testing images.")
            
            # Extract label targets to compute Dirichlet distribution partition indices
            train_labels = np.array(train_dataset.targets)
            client_indices = partition_non_iid_dirichlet(train_labels, num_clients, alpha)
            
            client_loaders = []
            for k in range(num_clients):
                client_subset = Subset(train_dataset, client_indices[k])
                loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, drop_last=False)
                client_loaders.append(loader)
                
                # Print label stats
                assigned_labels = train_labels[client_indices[k]]
                unique, counts = np.unique(assigned_labels, return_counts=True)
                distribution = dict(zip([EMOTIONS[u] for u in unique], counts.tolist()))
                print(f"Client {k} allocated {len(client_indices[k])} samples. Label distribution: {distribution}")
                
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return client_loaders, test_loader
            
        except Exception as e:
            print(f"[DATA] Failed loading from ImageFolder: {e}. Falling back to CSV/Synthetic...")
            
    # Fallback to CSV / Synthetic loader
    train_imgs, train_lbls, test_imgs, test_lbls = load_fer2013(csv_path)
    
    train_dataset = FER2013Dataset(train_imgs, train_lbls)
    test_dataset = FER2013Dataset(test_imgs, test_lbls)
    
    client_indices = partition_non_iid_dirichlet(train_lbls, num_clients, alpha)
    
    client_loaders = []
    for k in range(num_clients):
        client_subset = Subset(train_dataset, client_indices[k])
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)
        
        assigned_labels = train_lbls[client_indices[k]]
        unique, counts = np.unique(assigned_labels, return_counts=True)
        distribution = dict(zip([EMOTIONS[u] for u in unique], counts.tolist()))
        print(f"Client {k} allocated {len(client_indices[k])} samples. Label distribution: {distribution}")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader
