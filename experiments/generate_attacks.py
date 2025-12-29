"""
Adversarial Attack Generator using PGD

Generates adversarial examples for the trained N=10 Parity MLP.
Uses Projected Gradient Descent (PGD) to find perturbations
that flip the model's predictions.

Collects ONLY successful attacks (continuous float values).
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Configuration
N = 10
HIDDEN_SIZE = 128
EPSILON = 0.5     # L-infinity perturbation bound
ALPHA = 0.05      # Step size
STEPS = 20        # PGD iterations per attempt
TARGET_SUCCESSES = 100
MAX_ATTEMPTS = 2000  # Give up after this many samples

class ParityMLP(nn.Module):
    """MLP for learning parity function (must match nn_probe.py)."""
    def __init__(self, input_size=10, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

def get_hypercube_data(n):
    """Generate all 2^n binary vectors and parity labels."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    X = ((indices[:, None] & masks) > 0).astype(np.float32)
    y = (X.sum(axis=1) % 2).astype(np.float32)
    return X, y

def pgd_attack(model, x, y_true, epsilon=EPSILON, alpha=ALPHA, steps=STEPS):
    """
    Projected Gradient Descent attack.
    
    Args:
        model: Target model
        x: Clean input (1, n)
        y_true: True label
        epsilon: Max perturbation (L-infinity)
        alpha: Step size
        steps: Number of iterations
    
    Returns:
        x_adv: Adversarial example (continuous floats)
        success: Whether attack flipped prediction
    """
    x_orig = x.clone().detach()
    
    # Initialize with small random perturbation
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for step in range(steps):
        x_adv.requires_grad_(True)
        
        output = model(x_adv)
        pred_binary = (output > 0.5).float()
        
        # Check success - prediction flipped
        if pred_binary.item() != y_true.item():
            return x_adv.detach(), True
        
        # Compute loss to maximize error (minimize confidence in true class)
        if y_true.item() == 1:
            loss = output  # Push toward 0
        else:
            loss = 1 - output  # Push toward 1
        
        # Backprop
        model.zero_grad()
        loss.backward()
        
        # PGD step: move in direction of gradient
        grad = x_adv.grad
        x_adv = x_adv.detach() + alpha * grad.sign()
        
        # Project back to epsilon-ball around original
        perturbation = x_adv - x_orig
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        x_adv = x_orig + perturbation
        
        # Clip to valid input range [0, 1]
        x_adv = torch.clamp(x_adv, 0, 1)
    
    # Final check
    with torch.no_grad():
        output = model(x_adv)
        pred_binary = (output > 0.5).float()
        success = pred_binary.item() != y_true.item()
    
    return x_adv.detach(), success

def load_model():
    """Load trained model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'parity_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run nn_probe.py first.")
    
    model = ParityMLP(input_size=N, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    return model

def generate_adversarial_samples():
    print("=" * 60)
    print("PGD ADVERSARIAL ATTACK GENERATOR")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  N = {N}")
    print(f"  Epsilon = {EPSILON}")
    print(f"  Alpha (step size) = {ALPHA}")
    print(f"  Steps = {STEPS}")
    print(f"  Target successes = {TARGET_SUCCESSES}")
    
    # Load model
    model = load_model()
    print(f"\nModel loaded successfully")
    
    # Get data
    X_np, y_np = get_hypercube_data(N)
    X = torch.tensor(X_np)
    y = torch.tensor(y_np)
    
    # Verify model accuracy
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        accuracy = (preds == y).float().mean().item()
    print(f"Model accuracy on clean data: {accuracy:.4f}")
    
    # Find correctly classified samples
    correct_mask = (preds == y).numpy()
    correct_indices = np.where(correct_mask)[0]
    np.random.seed(42)
    np.random.shuffle(correct_indices)
    
    print(f"\nGenerating adversarial examples...")
    print(f"{'Attempts':<12} | {'Successes':<12} | {'Rate':<10}")
    print("-" * 40)
    
    clean_samples = []
    adversarial_samples = []
    attempts = 0
    
    for idx in correct_indices:
        if len(adversarial_samples) >= TARGET_SUCCESSES:
            break
        if attempts >= MAX_ATTEMPTS:
            break
        
        x_clean = X[idx:idx+1]
        y_true = y[idx:idx+1]
        
        x_adv, success = pgd_attack(model, x_clean, y_true)
        attempts += 1
        
        if success:
            clean_samples.append(x_clean.numpy().flatten())
            adversarial_samples.append(x_adv.numpy().flatten())
        
        if attempts % 100 == 0:
            rate = len(adversarial_samples) / attempts if attempts > 0 else 0
            print(f"{attempts:<12} | {len(adversarial_samples):<12} | {rate:.2%}")
    
    # Final stats
    rate = len(adversarial_samples) / attempts if attempts > 0 else 0
    print(f"{attempts:<12} | {len(adversarial_samples):<12} | {rate:.2%}")
    
    if len(adversarial_samples) == 0:
        print("\n⚠ No successful attacks! Try increasing epsilon or steps.")
        return
    
    # Save results (continuous floats, not rounded)
    clean_np = np.array(clean_samples)
    adv_np = np.array(adversarial_samples)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'adversarial_data.npz')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez_compressed(output_path, 
                        clean_samples=clean_np, 
                        adversarial_samples=adv_np)
    
    print(f"\n✓ Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    print(f"Successful attacks: {len(adversarial_samples)}")
    print(f"Attempts needed: {attempts}")
    print(f"Success rate: {len(adversarial_samples)/attempts:.1%}")
    
    # Perturbation stats
    perturbations = adv_np - clean_np
    print(f"\nPerturbation Statistics:")
    print(f"  L-infinity (max): {np.max(np.abs(perturbations)):.4f}")
    print(f"  L-infinity (avg): {np.mean(np.max(np.abs(perturbations), axis=1)):.4f}")
    print(f"  L2 (avg): {np.mean(np.linalg.norm(perturbations, axis=1)):.4f}")
    
    # Show example
    print(f"\nExample adversarial perturbation:")
    print(f"  Clean:      {clean_np[0]}")
    print(f"  Adversarial: {np.round(adv_np[0], 3)}")
    print(f"  Difference:  {np.round(perturbations[0], 3)}")

if __name__ == "__main__":
    generate_adversarial_samples()
