"""
MNIST Manifold Evolution Visualization

Uses t-SNE to project latent space to 2D, comparing epoch 0 vs epoch 10.
Shows that Low Topological Complexity = High Semantic Order.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import ripser
import matplotlib.pyplot as plt

# Configuration
LATENT_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 10
N_SAMPLES = 1000

class SimpleCNN(nn.Module):
    """Simple CNN with extractable latent space."""
    def __init__(self, latent_size=LATENT_SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(64 * 7 * 7, latent_size)
        self.output = nn.Linear(latent_size, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        latent = self.latent(x)
        out = self.output(torch.relu(latent))
        return out
    
    def get_latent(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.latent(x)

def add_jitter(points, sigma=0.02):
    return points + np.random.normal(0, sigma, points.shape)

def compute_pcc_binary(activations, max_samples=500):
    binary = (activations > 0).astype(np.float32)
    if len(binary) < 5:
        return 0.0
    if len(binary) > max_samples:
        indices = np.random.choice(len(binary), max_samples, replace=False)
        binary = binary[indices]
    pts = add_jitter(binary)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
    dgms = results['dgms']
    pcc = 0.0
    for dim in range(1, len(dgms)):
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            if len(lifetimes) > 0:
                pcc += np.sum(lifetimes ** 2)
    return pcc

def extract_latent(model, data_loader, n_samples):
    model.eval()
    activations, labels = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            latent = model.get_latent(images)
            activations.append(latent.numpy())
            labels.append(targets.numpy())
            if sum(len(a) for a in activations) >= n_samples:
                break
    return np.concatenate(activations)[:n_samples], np.concatenate(labels)[:n_samples]

def run_visualization():
    print("=" * 60)
    print("MNIST MANIFOLD EVOLUTION")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleCNN(LATENT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Epoch 0
    print("\n[1] Extracting Epoch 0 latent vectors...")
    latent_0, labels_0 = extract_latent(model, test_loader, N_SAMPLES)
    pcc_0 = compute_pcc_binary(latent_0)
    print(f"    PCC: {pcc_0:.4f}")
    
    # Train
    print(f"\n[2] Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"    Epoch {epoch}/{EPOCHS} complete")
    
    # Epoch 10
    print("\n[3] Extracting Epoch 10 latent vectors...")
    latent_10, labels_10 = extract_latent(model, test_loader, N_SAMPLES)
    pcc_10 = compute_pcc_binary(latent_10)
    print(f"    PCC: {pcc_10:.4f}")
    
    # t-SNE
    print("\n[4] Running t-SNE projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    proj_0 = tsne.fit_transform(latent_0)
    proj_10 = tsne.fit_transform(latent_10)
    
    # Plot
    print("\n[5] Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cmap = plt.cm.get_cmap('tab10', 10)
    
    # Epoch 0
    ax = axes[0]
    for digit in range(10):
        mask = labels_0 == digit
        ax.scatter(proj_0[mask, 0], proj_0[mask, 1], 
                   c=[cmap(digit)], label=str(digit), alpha=0.6, s=20)
    ax.set_title(f"Epoch 0: Untrained Network\nPCC = {pcc_0:.2f} (High Complexity)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(title='Digit', loc='upper right', fontsize=8)
    ax.text(0.02, 0.98, 'Random Fog\n(No Structure)', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Epoch 10
    ax = axes[1]
    for digit in range(10):
        mask = labels_10 == digit
        ax.scatter(proj_10[mask, 0], proj_10[mask, 1], 
                   c=[cmap(digit)], label=str(digit), alpha=0.6, s=20)
    ax.set_title(f"Epoch 10: Trained Network\nPCC = {pcc_10:.2f} (Lower Complexity)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(title='Digit', loc='upper right', fontsize=8)
    ax.text(0.02, 0.98, 'Digit Clusters\n(Semantic Order)', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    fig.suptitle("The Mind of the Machine: Latent Space Evolution on MNIST\n" +
                 "Low Topological Complexity = High Semantic Order",
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'mnist_manifold_evolution.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Epoch 0 PCC:  {pcc_0:.4f} (Random)")
    print(f"  Epoch 10 PCC: {pcc_10:.4f} (Organized)")
    print(f"  Reduction:    {pcc_0 - pcc_10:.4f} ({100*(pcc_0-pcc_10)/pcc_0:.1f}%)")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_visualization()
