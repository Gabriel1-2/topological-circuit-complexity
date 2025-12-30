"""
MNIST Neural Topology Analysis

Analyzes the topological structure of neural network latent representations
as training progresses. Hypothesis: Random fog (high PCC) -> Clean clusters (low PCC).
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
LATENT_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 10
N_TEST_SAMPLES = 1000
MAX_PCC_SAMPLES = 500

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
        """Extract latent space activations."""
        x = self.conv(x)
        x = self.flatten(x)
        latent = self.latent(x)
        return latent

def add_jitter(points, sigma=0.02):
    """Add jitter for TDA stability."""
    return points + np.random.normal(0, sigma, points.shape)

def compute_pcc_binary(activations, max_samples=MAX_PCC_SAMPLES):
    """
    Compute PCC of binary activation patterns.
    Active > 0, Inactive <= 0
    """
    # Binarize
    binary = (activations > 0).astype(np.float32)
    
    if len(binary) < 5:
        return 0.0
    
    # Subsample
    if len(binary) > max_samples:
        indices = np.random.choice(len(binary), max_samples, replace=False)
        binary = binary[indices]
    
    # Jitter and compute distance
    pts = add_jitter(binary)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    # Ripser
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

def extract_latent_activations(model, data_loader, n_samples=N_TEST_SAMPLES):
    """Extract latent space activations for n samples."""
    model.eval()
    activations = []
    labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            latent = model.get_latent(images)
            activations.append(latent.numpy())
            labels.append(targets.numpy())
            
            if sum(len(a) for a in activations) >= n_samples:
                break
    
    activations = np.concatenate(activations)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]
    
    return activations, labels

def run_experiment():
    print("=" * 60)
    print("MNIST NEURAL TOPOLOGY ANALYSIS")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Latent size: {LATENT_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Test samples: {N_TEST_SAMPLES}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nLoaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Model
    model = SimpleCNN(LATENT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    results = []
    
    # Initial state (epoch 0)
    print(f"\n[Epoch 0] Analyzing untrained network...")
    activations, _ = extract_latent_activations(model, test_loader)
    pcc = compute_pcc_binary(activations)
    
    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = correct / total
    
    print(f"  Accuracy: {accuracy:.4f}, PCC: {pcc:.4f}")
    results.append({'epoch': 0, 'accuracy': accuracy, 'pcc': pcc})
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = correct / total
        
        # TDA on latent space
        activations, _ = extract_latent_activations(model, test_loader)
        pcc = compute_pcc_binary(activations)
        
        print(f"[Epoch {epoch}] Accuracy: {accuracy:.4f}, PCC: {pcc:.4f}")
        results.append({'epoch': epoch, 'accuracy': accuracy, 'pcc': pcc})
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_topology.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'accuracy', 'pcc'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TOPOLOGY EVOLUTION SUMMARY")
    print("=" * 60)
    
    initial_pcc = results[0]['pcc']
    final_pcc = results[-1]['pcc']
    pcc_change = final_pcc - initial_pcc
    
    print(f"\n  Initial PCC (Epoch 0):  {initial_pcc:.4f}")
    print(f"  Final PCC (Epoch {EPOCHS}): {final_pcc:.4f}")
    print(f"  Change:                 {pcc_change:+.4f}")
    
    if pcc_change < 0:
        print("\n  ✓ HYPOTHESIS CONFIRMED: Training REDUCED latent space complexity!")
        print("    Random fog → Clean clusters")
    else:
        print("\n  ✗ HYPOTHESIS REJECTED: Training INCREASED complexity.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_experiment()
