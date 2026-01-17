
import sys
import os
import pickle
import numpy as np
import pandas as pd
from MiCoGPT.utils_vCross.corpus_vCross import MiCoGPTCorpusVCross

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    data_path = "/Users/kiancai/STA24/CWD/STAi/MiCoGPT/data/vCross/ResMicroDB_90338_vCross.pkl"
    print(f"Loading corpus from {data_path}...")
    
    with open(data_path, "rb") as f:
        corpus = pickle.load(f)
        
    print(f"Corpus loaded. Number of samples: {len(corpus)}")
    print(f"Number of bins: {corpus.num_bins}")
    print(f"Normalize total: {corpus.normalize_total}")
    print(f"Log1p: {corpus.log1p}")
    
    # Pick the first sample
    sample_id = corpus.sample_ids[0]
    print(f"\nAnalyzing sample: {sample_id}")
    
    # Get preprocessed data (already normalized and log1p-ed)
    sample_data = corpus.data.loc[sample_id]
    
    # Filter zeros
    sample_values = sample_data[sample_data > 0]
    print(f"Number of non-zero taxa: {len(sample_values)}")
    
    # Calculate quantiles as in the code
    # q = np.linspace(0, 1, self.num_bins - 1)
    q = np.linspace(0, 1, corpus.num_bins - 1)
    bin_edges = np.quantile(sample_values, q)
    
    print("\nBin Edges (in Log1p + Normalized space):")
    # Print a few edges
    print(f"Bin 1 upper bound (0%): {bin_edges[0]:.4f}")
    print(f"Bin 25 upper bound (approx median): {bin_edges[len(bin_edges)//2]:.4f}")
    print(f"Bin {corpus.num_bins-1} upper bound (100%): {bin_edges[-1]:.4f}")
    
    # Inverse transform to get relative abundance
    # 1. Inverse Log1p
    edges_exp = np.expm1(bin_edges)
    
    # 2. Inverse Normalization
    # Since relative abundance = count / library_size
    # And preprocessed = (count / library_size) * normalize_total
    # So relative abundance = preprocessed / normalize_total (approx, ignoring log1p for a moment)
    # Actually:
    # x_norm = (x_raw / lib_size) * target_sum
    # x_final = log1p(x_norm)
    # So:
    # x_norm = expm1(x_final)
    # rel_abundance = x_raw / lib_size = x_norm / target_sum
    
    target_sum = corpus.normalize_total if corpus.normalize_total else 10000 # Default fallback if None, but should check if it's stored
    # Wait, if normalize_total is None in __init__, it calculates median and uses it, but DOES IT STORE IT?
    # In the code: self.normalize_total = normalize_total. It stores the *argument*.
    # But the median is calculated inside _preprocess and not stored back to self.normalize_total.
    # However, since self.data IS already processed, we can infer target_sum if we assume the rows sum to roughly target_sum (before log1p).
    # But after log1p, the sum is different.
    
    # Let's check the sum of expm1 of the sample data
    inferred_sum = np.expm1(sample_values).sum()
    print(f"\nInferred normalization target sum (sum of expm1(values)): {inferred_sum:.2f}")
    
    target_sum = inferred_sum # Use this for conversion
    
    edges_rel = edges_exp / target_sum
    
    print("\nApproximate Relative Abundance Cutoffs:")
    print(f"Min (Bin 1 start): {edges_rel[0]:.2e}") # Actually edges[0] is min value
    print(f"Median (Bin {corpus.num_bins//2}): {edges_rel[len(edges_rel)//2]:.2e}")
    print(f"Max (Bin {corpus.num_bins}): {edges_rel[-1]:.2e}")
    
    # Show all edges nicely
    print("\nDetailed Mapping (Sample specific):")
    for i in range(0, len(edges_rel), 5):
        print(f"Bin {i+1} edge: {edges_rel[i]:.6f} (Rank {q[i]*100:.1f}%)")

if __name__ == "__main__":
    main()
