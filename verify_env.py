#!/usr/bin/env python3
"""
Topological Circuit Complexity - Environment Verification Script

This script verifies that all TDA dependencies are correctly installed,
with special focus on confirming the C++ bindings for ripser are working.
"""

import sys

def main():
    print("=" * 60)
    print("Topological Circuit Complexity - Environment Verification")
    print("=" * 60)
    
    errors = []
    
    # Test numpy
    print("\n[1/6] Testing numpy...", end=" ")
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        errors.append("numpy")
    
    # Test scikit-learn
    print("[2/6] Testing scikit-learn...", end=" ")
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        errors.append("scikit-learn")
    
    # Test networkx
    print("[3/6] Testing networkx...", end=" ")
    try:
        import networkx as nx
        print(f"✓ networkx {nx.__version__}")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        errors.append("networkx")
    
    # Test gudhi
    print("[4/6] Testing gudhi...", end=" ")
    try:
        import gudhi
        print(f"✓ gudhi {gudhi.__version__}")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        errors.append("gudhi")
    
    # Test persim
    print("[5/6] Testing persim...", end=" ")
    try:
        import persim
        print(f"✓ persim {persim.__version__}")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        errors.append("persim")
    
    # Test ripser with actual computation (C++ bindings verification)
    print("[6/6] Testing ripser (C++ bindings)...", end=" ")
    try:
        from ripser import ripser
        import numpy as np
        
        # Generate random 10-point cloud in 3D
        np.random.seed(42)
        point_cloud = np.random.rand(10, 3)
        
        # Compute persistent homology (this exercises the C++ backend)
        result = ripser(point_cloud, maxdim=1)
        dgms = result['dgms']
        
        # Validate output structure
        assert len(dgms) >= 2, "Expected at least H0 and H1 diagrams"
        h0_points = len(dgms[0])
        h1_points = len(dgms[1])
        
        print(f"✓ ripser working!")
        print(f"    └─ Persistence computation successful:")
        print(f"       • H0 (components): {h0_points} features")
        print(f"       • H1 (loops): {h1_points} features")
        
    except ImportError as e:
        print(f"✗ FAILED to import: {e}")
        errors.append("ripser")
    except Exception as e:
        print(f"✗ FAILED during computation: {e}")
        errors.append("ripser-computation")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"VERIFICATION FAILED - {len(errors)} error(s): {', '.join(errors)}")
        print("Please check your installation and try again.")
        sys.exit(1)
    else:
        print("✓ ALL CHECKS PASSED - Environment is ready!")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
