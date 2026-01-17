
import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MiCoGPT.utils_vCross.corpus_vCross import MiCoGPTCorpusVCross

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    data_path = "/Users/kiancai/STA24/CWD/STAi/MiCoGPT/data/vCross/ResMicroDB_90338_vCross.pkl"
    print(f"Loading corpus from {data_path}...")
    
    with open(data_path, "rb") as f:
        corpus = pickle.load(f)
        
    print(f"Corpus loaded. Num samples: {len(corpus)}")
    
    # Select 3 distinct samples to compare
    # We want to see if the relationship between Rank (Position) and Bin ID varies across samples
    
    # Let's pick samples with different "richness" (number of non-zero taxa)
    # Since we can't easily query richness without iterating, we'll just pick first 100, calculate richness, and pick 3 representative ones.
    
    samples_stats = []
    print("Scanning first 100 samples for diversity...")
    for i in range(100):
        sid = corpus.sample_ids[i]
        sdata = corpus.data.loc[sid]
        sdata = sdata[sdata > 0]
        samples_stats.append({
            'id': sid,
            'richness': len(sdata),
            'max_val': sdata.max(),
            'data': sdata
        })
    
    # Sort by richness
    samples_stats.sort(key=lambda x: x['richness'])
    
    # Pick Low, Medium, High richness
    selected = [
        samples_stats[0], # Min richness
        samples_stats[len(samples_stats)//2], # Median
        samples_stats[-1] # Max richness
    ]
    
    print("\nComparing Rank vs Bin ID for 3 samples:")
    
    for s in selected:
        print(f"\nSample: {s['id']} (Richness: {s['richness']})")
        
        # Simulate the binning process
        sample_data = s['data']
        # Already normalized and log1p-ed in corpus
        
        # Sort descending (Rank)
        sample_sorted = sample_data.sort_values(ascending=False)
        
        # Calculate Bins (Dynamic)
        q = np.linspace(0, 1, corpus.num_bins - 1)
        bins = np.quantile(sample_sorted.values, q)
        
        # Digitize
        # We use simple digitize here for visualization, ignoring the random seed perturbation for simplicity
        # or we can use the corpus method if we could instantiate it, but let's just approximate
        # digitize returns 0..num_bins-1. 
        bin_ids = np.digitize(sample_sorted.values, bins) + 1
        
        # Output the curve data points
        # Format: Position(Rank) -> BinID
        print("Position(x) | Value(y) | BinID(z)")
        print("-" * 30)
        
        # Print every k-th point to keep log short
        step = max(1, len(sample_sorted) // 10)
        for i in range(0, len(sample_sorted), step):
            val = sample_sorted.iloc[i]
            bid = bin_ids[i]
            print(f"{i:4d}        | {val:.4f}   | {bid}")
            
        # Also print the last one
        print(f"{len(sample_sorted)-1:4d}        | {sample_sorted.iloc[-1]:.4f}   | {bin_ids[-1]}")

if __name__ == "__main__":
    main()
