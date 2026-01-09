import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter



def analyze_protein_sequences(protein_sequences):
    """
    Analyze protein sequence lengths and create distribution plots
    
    Parameters:
    protein_sequences: list of protein sequences (strings)
    """
    
    # Calculate sequence lengths
    sequence_lengths = [len(seq) for seq in protein_sequences]
    
    # Statistical analysis
    print("=== Protein Sequence Length Analysis ===")
    print(f"Total sequences: {len(protein_sequences)}")
    print(f"Mean length: {np.mean(sequence_lengths):.2f}")
    print(f"Median length: {np.median(sequence_lengths):.2f}")
    print(f"Min length: {np.min(sequence_lengths)}")
    print(f"Max length: {np.max(sequence_lengths)}")
    print(f"Standard deviation: {np.std(sequence_lengths):.2f}")
    print(f"25th percentile: {np.percentile(sequence_lengths, 25):.2f}")
    print(f"75th percentile: {np.percentile(sequence_lengths, 75):.2f}")
    
    # Create comprehensive plots (will be saved automatically)
    # Plots will be created in analyze_protein_sequences when called from main
    
    # Return analysis results
    return {
        'lengths': sequence_lengths,
        'statistics': {
            'total_sequences': len(protein_sequences),
            'mean_length': np.mean(sequence_lengths),
            'median_length': np.median(sequence_lengths),
            'min_length': np.min(sequence_lengths),
            'max_length': np.max(sequence_lengths),
            'std_length': np.std(sequence_lengths)
        }
    }

def create_distribution_plots(lengths, output_dir='./plots'):
    """
    Create comprehensive distribution plots with English labels
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font that supports English
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(28, 12))
    fig.suptitle('Protein Sequence Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Basic histogram
    axes[0,0].hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Sequence Length')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Sequence Length Distribution')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Density plot
    from scipy.stats import gaussian_kde
    density = gaussian_kde(lengths)
    xs = np.linspace(min(lengths), max(lengths), 200)
    axes[0,1].plot(xs, density(xs), 'r-', linewidth=2)
    axes[0,1].fill_between(xs, density(xs), alpha=0.3, color='red')
    axes[0,1].set_xlabel('Sequence Length')
    axes[0,1].set_ylabel('Probability Density')
    axes[0,1].set_title('Length Distribution Density')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Box plot
    axes[0,2].boxplot(lengths, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', color='black'),
                     medianprops=dict(color='red', linewidth=2))
    axes[0,2].set_ylabel('Sequence Length')
    axes[0,2].set_title('Length Distribution Box Plot')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_lengths = np.sort(lengths)
    yvals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths))
    axes[1,0].plot(sorted_lengths, yvals, 'g-', linewidth=2)
    axes[1,0].set_xlabel('Sequence Length')
    axes[1,0].set_ylabel('Cumulative Probability')
    axes[1,0].set_title('Cumulative Distribution Function')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Histogram with log scale
    axes[1,1].hist(lengths, bins=50, alpha=0.7, color='orange', 
                  edgecolor='black', log=True)
    axes[1,1].set_xlabel('Sequence Length')
    axes[1,1].set_ylabel('Frequency (log scale)')
    axes[1,1].set_title('Length Distribution (Log Scale)')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Length bin statistics
    length_bins = [0, 100, 200, 300, 500, 1000, max(lengths)]
    bin_labels = ['0-100', '101-200', '201-300', '301-500', '501-1000', '>1000']
    binned_lengths = pd.cut(lengths, bins=length_bins, labels=bin_labels)
    bin_counts = binned_lengths.value_counts().sort_index()
    
    axes[1,2].bar(bin_labels, bin_counts.values, color='lightcoral', alpha=0.7)
    axes[1,2].set_xlabel('Length Range')
    axes[1,2].set_ylabel('Number of Sequences')
    axes[1,2].set_title('Distribution by Length Range')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(bin_counts.values):
        axes[1,2].text(i, count + max(bin_counts.values)*0.01, str(count), 
                      ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_distribution_plots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comprehensive_distribution_plots.png")
    plt.close()
    
    # Create detailed statistical plot
    create_detailed_statistics_plot(lengths, output_dir)

def create_detailed_statistics_plot(lengths, output_dir='./plots'):
    """
    Create a detailed statistical plot with English labels
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use seaborn for a nicer distribution plot
    sns.histplot(lengths, kde=True, ax=ax, color='steelblue', 
                alpha=0.7, edgecolor='black')
    
    # Add statistical lines
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    ax.axvline(mean_length, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_length:.1f}')
    ax.axvline(median_length, color='green', linestyle='--', linewidth=2, 
               label=f'Median: {median_length:.1f}')
    
    ax.set_xlabel('Protein Sequence Length', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Protein Sequence Length Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistical text box
    stats_text = f'''
    Statistical Summary:
    Total Sequences: {len(lengths)}
    Min Length: {np.min(lengths)}
    Max Length: {np.max(lengths)}
    Standard Deviation: {np.std(lengths):.1f}
    '''
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_statistics_plot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/detailed_statistics_plot.png")
    plt.close()

def export_analysis_results(protein_sequences, output_file='protein_analysis_results.csv'):
    """
    Export analysis results to CSV file
    """
    lengths = [len(seq) for seq in protein_sequences]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sequence_index': range(1, len(protein_sequences) + 1),
        'sequence_length': lengths,
        'sequence_preview': [seq[:50] + '...' if len(seq) > 50 else seq for seq in protein_sequences]
    })
    
    # Add length classification
    def classify_length(length):
        if length <= 100:
            return 'Very Short (≤100)'
        elif length <= 300:
            return 'Short (101-300)'
        elif length <= 600:
            return 'Medium (301-600)'
        elif length <= 1000:
            return 'Long (601-1000)'
        else:
            return 'Very Long (>1000)'
    
    results_df['length_category'] = results_df['sequence_length'].apply(classify_length)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Analysis results exported to: {output_file}")
    
    return results_df

# Simplified version for quick analysis
def simple_length_analysis(protein_sequences, output_dir='./plots'):
    """
    Simple protein sequence length analysis with basic plot
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    # Calculate lengths
    lengths = [len(seq) for seq in protein_sequences]
    
    # Basic statistics
    print(f"Number of sequences: {len(lengths)}")
    print(f"Average length: {np.mean(lengths):.1f}")
    print(f"Length range: {min(lengths)} - {max(lengths)}")
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(lengths, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    
    plt.xlabel('Protein Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Protein Sequence Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/simple_length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/simple_length_analysis.png")
    plt.close()
    
    return lengths

# # Usage example
# if __name__ == "__main__":
#     # Example protein sequences - replace with your actual data
#     example_protein_sequences = [
#         "MAGWHKYTLS",  # Short sequence example
#         "MKALIVLGGVGAGLSAAAILVAMFVFF",  # Medium length example
#         "MKTIIALSYIFCLVFADYKDDDDKHHHHHH",  # Longer sequence
#         "MPEPTIDE" * 50,  # Long sequence example
#         "MAS" * 100,  # Very long sequence example
#         # ... Add all your protein sequences here
#     ]
    
#     # Analyze sequences
#     analysis_results = analyze_protein_sequences(example_protein_sequences)
    
#     # Export results
#     results_df = export_analysis_results(example_protein_sequences)
    
#     print("\n=== Analysis Complete ===")
#     print(f"Analyzed {len(example_protein_sequences)} protein sequences")

def read_fasta(fasta_file):
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()[1:]
                fasta_dict[seq_name] = ""
            else:
                fasta_dict[seq_name] += line.strip()
    fasta_list = [(seq_name, seq) for seq_name, seq in fasta_dict.items()]
    return fasta_list

#---------
if __name__ == "__main__":
    import os
    
    fasta_path = "/home_zfs/wanghanyi/mRNAHyena/data/mRNA_data/NM_vertebrate_mammalian_mRNA_filtered_separators.fasta"
    output_dir = "./demo/sequence_analysis_plots"
    
    print(f"Reading sequences from: {fasta_path}")
    fasta_data = read_fasta(fasta_path)
    # 提取纯序列列表
    Sequence_list = [seq for _, seq in fasta_data]
    print(f"Loaded {len(Sequence_list)} sequences\n")
    
    # 统计分析（不画图）
    sequence_lengths = [len(seq) for seq in Sequence_list]
    print("=== Sequence Length Analysis ===")
    print(f"Total sequences: {len(Sequence_list)}")
    print(f"Mean length: {np.mean(sequence_lengths):.2f}")
    print(f"Median length: {np.median(sequence_lengths):.2f}")
    print(f"Min length: {np.min(sequence_lengths)}")
    print(f"Max length: {np.max(sequence_lengths)}")
    print(f"Standard deviation: {np.std(sequence_lengths):.2f}")
    print(f"25th percentile: {np.percentile(sequence_lengths, 25):.2f}")
    print(f"75th percentile: {np.percentile(sequence_lengths, 75):.2f}")
    print()
    
    # 生成图表
    print(f"Generating plots in: {output_dir}")
    create_distribution_plots(sequence_lengths, output_dir)
    
    # 导出 CSV（可选）
    # export_analysis_results(Sequence_list, output_file="protein_analysis_results.csv")
    
    print("\n=== Analysis Complete ===")