import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

EMOTIONS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Canonical label order (FER-2013 standard used throughout this project)
CLASS_TO_IDX = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "sad": 4, "surprise": 5, "neutral": 6
}

# Shared transforms (PIL-based, same for CSV and ImageFolder paths)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    # Matches reference: rotation 10°, translate 10%, zoom ±10%
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class FER2013Dataset(Dataset):
    """Dataset wrapping numpy image arrays loaded from the FER-2013 CSV."""
    def __init__(self, images, labels, transform=None):
        # images: numpy (N, H, W) float32 in [0, 1]
        self.images    = images
        self.labels    = labels
        self.transform = transform or TEST_TRANSFORM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]  # (H, W) float32 [0, 1]
        label = int(self.labels[idx])
        # Convert to uint8 PIL L (grayscale) so torchvision transforms work identically
        # to the ImageFolder path
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        return self.transform(pil_img), torch.tensor(label, dtype=torch.long)


def load_fer2013_csv(csv_path: str):
    """
    Load FER-2013 from the competition CSV (pixel strings + emotion labels).
    Returns numpy arrays: train_imgs, train_lbls, test_imgs, test_lbls.
    """
    print(f"[DATA] Loading FER-2013 from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    labels = df['emotion'].values.astype(np.int64)
    images = np.array(
        [np.fromstring(p, dtype=np.uint8, sep=' ').reshape(48, 48) / 255.0
         for p in df['pixels'].values],
        dtype=np.float32
    )
    if 'Usage' in df.columns:
        train_mask = df['Usage'] == 'Training'
        test_mask  = df['Usage'].isin(['PublicTest', 'PrivateTest'])
        return images[train_mask], labels[train_mask], images[test_mask], labels[test_mask]
    split = int(0.8 * len(images))
    return images[:split], labels[:split], images[split:], labels[split:]


def generate_synthetic_fer(num_samples=10000):
    print(f"[DATA] Generating {num_samples} synthetic 48×48 images (fallback)...")
    images = np.random.randint(0, 256, size=(num_samples, 48, 48), dtype=np.uint8) / 255.0
    labels = np.random.choice(7, size=num_samples,
                               p=[0.13, 0.02, 0.14, 0.25, 0.17, 0.11, 0.18])
    return images.astype(np.float32), labels.astype(np.int64)


def partition_non_iid_dirichlet(labels, num_clients, alpha=0.5):
    num_classes = 7
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = np.round(proportions * len(idx_c)).astype(int)
        proportions[-1] += len(idx_c) - proportions.sum()
        start = 0
        for k in range(num_clients):
            end = start + proportions[k]
            client_indices[k].extend(idx_c[start:end])
            start = end
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
    return client_indices


def _build_loaders(train_dataset, test_dataset, train_labels,
                   num_clients, alpha, batch_size, max_samples_per_client):
    """Partition training data across clients and return (client_loaders, test_loader)."""
    client_indices = partition_non_iid_dirichlet(train_labels, num_clients, alpha)
    client_loaders = []
    for k in range(num_clients):
        idx = client_indices[k]
        if max_samples_per_client and len(idx) > max_samples_per_client:
            idx = idx[:max_samples_per_client]
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=0, pin_memory=False)
        client_loaders.append(loader)
        assigned = train_labels[np.array(idx)]
        unique, counts = np.unique(assigned, return_counts=True)
        dist = {EMOTIONS[int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"  Client {k}: {len(idx)} samples — {dist}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return client_loaders, test_loader


def get_federated_loaders(csv_path=None, num_clients=5, alpha=0.5,
                          batch_size=32, max_samples_per_client=None):
    """
    Priority order:
      1. FER-2013 CSV (csv_path provided and exists) — authoritative competition data
      2. Kaggle image directories (data/train, data/test) — same data, different format
      3. Synthetic fallback
    All paths resize to 64×64 and apply the same augmentation pipeline.
    """

    # ── Priority 1: CSV ───────────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        try:
            train_imgs, train_lbls, test_imgs, test_lbls = load_fer2013_csv(csv_path)
            print(f"[DATA] Loaded {len(train_imgs)} train / {len(test_imgs)} test images from CSV.")
            train_dataset = FER2013Dataset(train_imgs, train_lbls, transform=TRAIN_TRANSFORM)
            test_dataset  = FER2013Dataset(test_imgs,  test_lbls,  transform=TEST_TRANSFORM)
            return _build_loaders(train_dataset, test_dataset, train_lbls,
                                  num_clients, alpha, batch_size, max_samples_per_client)
        except Exception as e:
            print(f"[DATA] CSV load failed: {e}. Falling back to ImageFolder...")

    # ── Priority 2: Kaggle image directories ─────────────────────────────────
    data_dir  = "data"
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("[DATA] Found Kaggle FER-2013 image directories.")
        try:
            train_dataset = ImageFolder(root=train_dir, transform=TRAIN_TRANSFORM)
            test_dataset  = ImageFolder(root=test_dir,  transform=TEST_TRANSFORM)

            # ImageFolder sorts class folders alphabetically:
            # angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
            # Remap to canonical order: neutral=6, sad=4, surprise=5
            alpha_idx = train_dataset.class_to_idx
            remap = {alpha_idx[cls]: CLASS_TO_IDX[cls] for cls in CLASS_TO_IDX if cls in alpha_idx}

            train_dataset.targets = [remap.get(t, t) for t in train_dataset.targets]
            test_dataset.targets  = [remap.get(t, t) for t in test_dataset.targets]
            train_dataset.samples = [(p, remap.get(l, l)) for p, l in train_dataset.samples]
            test_dataset.samples  = [(p, remap.get(l, l)) for p, l in test_dataset.samples]

            print(f"[DATA] Loaded {len(train_dataset)} train / {len(test_dataset)} test images.")
            print(f"[DATA] Label remap applied: {remap}")

            train_labels = np.array(train_dataset.targets)
            return _build_loaders(train_dataset, test_dataset, train_labels,
                                  num_clients, alpha, batch_size, max_samples_per_client)
        except Exception as e:
            print(f"[DATA] ImageFolder load failed: {e}. Falling back to synthetic data...")

    # ── Priority 3: Synthetic fallback ───────────────────────────────────────
    train_imgs, train_lbls = generate_synthetic_fer(8000)
    test_imgs,  test_lbls  = generate_synthetic_fer(2000)
    train_dataset = FER2013Dataset(train_imgs, train_lbls, transform=TRAIN_TRANSFORM)
    test_dataset  = FER2013Dataset(test_imgs,  test_lbls,  transform=TEST_TRANSFORM)
    return _build_loaders(train_dataset, test_dataset, train_lbls,
                          num_clients, alpha, batch_size, max_samples_per_client)
