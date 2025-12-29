
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from boolean_gen import generate_truth_table

class TestBooleanGen(unittest.TestCase):
    
    def test_majority_n3(self):
        """Test Majority function for N=3"""
        n = 3
        tt, coords = generate_truth_table(n, 'majority')
        
        # Hamming weights for 3 bits:
        # 0 (000) -> 0
        # 1 (001) -> 1
        # 2 (010) -> 1
        # 3 (011) -> 2 (Maj)
        # 4 (100) -> 1
        # 5 (101) -> 2 (Maj)
        # 6 (110) -> 2 (Maj)
        # 7 (111) -> 3 (Maj)
        # Threshold: > 1.5 => 2 or 3
        
        expected_indices = [3, 5, 6, 7] # Rows where weight >= 2
        
        # Check truth table values
        self.assertEqual(np.sum(tt), 4)
        for idx in expected_indices:
            self.assertEqual(tt[idx], 1, f"Index {idx} should be 1 for Majority(3)")
            
        # Check coordinates match
        # 3=011, 5=101, 6=110, 7=111
        expected_coords = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])
        
        np.testing.assert_array_equal(coords, expected_coords)
        
    def test_parity_n3(self):
        """Test Parity (Odd) function for N=3"""
        n = 3
        tt, coords = generate_truth_table(n, 'parity')
        
        # Odd weights: 1, 3
        # 1 (001) -> 1
        # 2 (010) -> 1
        # 4 (100) -> 1
        # 7 (111) -> 3
        
        expected_indices = [1, 2, 4, 7]
        
        self.assertEqual(np.sum(tt), 4)
        for idx in expected_indices:
            self.assertEqual(tt[idx], 1, f"Index {idx} should be 1 for Parity(3)")
            
        expected_coords = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ])
        
        np.testing.assert_array_equal(coords, expected_coords)

    def test_clique_n3_k3(self):
        """Test Clique function for N=3 (3 edges -> 3 vertices), k=3 (triangle)"""
        # N=3 bits represents edges of a 3-vertex graph.
        # Edges indices: (0,1), (0,2), (1,2)
        # Vertices: 0, 1, 2
        # A 3-clique requires ALL edges to be present.
        # So only 111 (index 7) should be 1.
        
        n = 3
        tt, coords = generate_truth_table(n, 'clique', k=3)
        
        self.assertEqual(np.sum(tt), 1)
        self.assertEqual(tt[7], 1)
        
        np.testing.assert_array_equal(coords, np.array([[1, 1, 1]]))

    def test_clique_validation_error(self):
        """Test that invalid N raises error for clique"""
        # N=4 is not valid for v(v-1)/2 
        # v=3 -> 3 edges
        # v=4 -> 6 edges
        # 4 is between.
        with self.assertRaises(ValueError):
            generate_truth_table(4, 'clique')

if __name__ == '__main__':
    unittest.main()
