"""
Visualize Self-Correction Results.
Creates a bar chart comparing Random Baseline, Majority Vote, HMM Selection, and Oracle.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_self_correction(input_file, output_dir):
    with open(input_file, "r") as f:
        data = json.load(f)
        
    # Data prep
    methods = ['Random', 'Majority Vote', 'HMM Check', 'Oracle']
    # Map keys from json to display names
    key_map = {
        'random': 'Random',
        'majority': 'Majority Vote',
        'hmm': 'HMM Check',
        'oracle': 'Oracle'
    }
    
    values = [data.get(k, 0) * 100 for k in ['random', 'majority', 'hmm', 'oracle']]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Colors: Grey for benchmarks, Blue for us, Green for Oracle
    colors = ['#95a5a6', '#95a5a6', '#3498db', '#2ecc71']
    
    bars = plt.bar(methods, values, color=colors, width=0.6)
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Self-Correction Performance (Best-of-N)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add numbers on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    # Add improvement arrow if HMM > Random
    if data['hmm'] > data['random']:
        imp = (data['hmm'] - data['random']) * 100
        plt.text(1, 90, f"+{imp:.1f}% Boost", color='#3498db', fontsize=12, fontweight='bold')
        
    plt.grid(axis='y', alpha=0.3)
    
    out_path = Path(output_dir) / "self_correction_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/self_correction/self_correction_results.json")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    
    plot_self_correction(args.input, args.output_dir)
