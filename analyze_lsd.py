"""
Calculate and visualize Log Spectral Distance (LSD) for V3 inference outputs.

This script:
1. Scans all folders in V3_outputs/
2. For each folder, calculates LSD between:
   - Target audio vs Direct prediction
   - Target audio vs Convolved prediction
3. Creates a comparative visualization
"""

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


def compute_log_spectral_distance(audio1: torch.Tensor, audio2: torch.Tensor, 
                                   sr: int = 44100, n_fft: int = 2048,
                                   hop_length: int = 512) -> float:
    """
    Compute Log Spectral Distance between two audio signals.
    
    LSD = sqrt(mean((log(S1) - log(S2))^2))
    
    Args:
        audio1: First audio tensor [channels, samples]
        audio2: Second audio tensor [channels, samples]
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        
    Returns:
        lsd: Log Spectral Distance in dB
    """
    # Convert to mono if stereo
    if audio1.shape[0] > 1:
        audio1 = torch.mean(audio1, dim=0, keepdim=True)
    if audio2.shape[0] > 1:
        audio2 = torch.mean(audio2, dim=0, keepdim=True)
    
    # Match lengths
    min_len = min(audio1.shape[-1], audio2.shape[-1])
    audio1 = audio1[..., :min_len]
    audio2 = audio2[..., :min_len]
    
    # Compute STFT
    stft1 = torch.stft(audio1.squeeze(0), n_fft=n_fft, hop_length=hop_length, 
                       return_complex=True, window=torch.hann_window(n_fft))
    stft2 = torch.stft(audio2.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                       return_complex=True, window=torch.hann_window(n_fft))
    
    # Get magnitude spectrograms
    mag1 = torch.abs(stft1)
    mag2 = torch.abs(stft2)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    mag1 = mag1 + eps
    mag2 = mag2 + eps
    
    # Compute log spectral distance
    log_diff = torch.log(mag1) - torch.log(mag2)
    lsd = torch.sqrt(torch.mean(log_diff ** 2))
    
    return lsd.item()


def analyze_folder(folder_path: Path) -> Dict[str, float]:
    """
    Analyze one V3_outputs subfolder and compute all LSD metrics.
    
    Args:
        folder_path: Path to folder containing inference outputs
        
    Returns:
        metrics: Dictionary of LSD values
    """
    metrics = {}
    
    # Define audio files to load
    files = {
        'target': folder_path / 'target.wav',
        'reference': folder_path / 'reference_synth.wav',
        'predicted_direct': folder_path / 'predicted_direct.wav',
        'predicted_convolved': folder_path / 'predicted_convolved.wav'
    }
    
    # Check if all files exist
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        print(f"⚠️  Skipping {folder_path.name}: missing files {missing}")
        return None
    
    # Load audio files
    audio = {}
    for name, path in files.items():
        waveform, sr = torchaudio.load(path)
        audio[name] = (waveform, sr)
    
    sr = audio['target'][1]  # Use target sample rate
    
    # Calculate LSD metrics
    print(f"📊 Analyzing {folder_path.name}...")
    
    # Target vs Predictions
    metrics['target_vs_direct'] = compute_log_spectral_distance(
        audio['target'][0], audio['predicted_direct'][0], sr
    )
    metrics['target_vs_convolved'] = compute_log_spectral_distance(
        audio['target'][0], audio['predicted_convolved'][0], sr
    )
    
    # Target vs Reference (baseline - how close was random reference to target)
    metrics['target_vs_reference'] = compute_log_spectral_distance(
        audio['target'][0], audio['reference'][0], sr
    )
    
    return metrics


def create_visualization(results: pd.DataFrame, output_path: Path):
    """
    Create comprehensive visualization of LSD metrics.
    
    Args:
        results: DataFrame with LSD metrics for all folders
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Log Spectral Distance: Target vs Predictions', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data
    x = np.arange(len(results))
    width = 0.25
    
    # Plot bars for each metric
    ax.bar(x - width, results['target_vs_direct'].values, width, 
           label='Target vs Direct', color='#FF6B6B', alpha=0.8)
    ax.bar(x, results['target_vs_convolved'].values, width,
           label='Target vs Convolved', color='#4ECDC4', alpha=0.8)
    ax.bar(x + width, results['target_vs_reference'].values, width,
           label='Target vs Reference (baseline)', color='#FFA07A', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Test Case', fontweight='bold', fontsize=12)
    ax.set_ylabel('Log Spectral Distance (dB)', fontweight='bold', fontsize=12)
    ax.set_title('Comparison of Prediction Methods', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results.index, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i in range(len(results)):
        direct = results['target_vs_direct'].values[i]
        convolved = results['target_vs_convolved'].values[i]
        reference = results['target_vs_reference'].values[i]
        
        ax.text(i - width, direct + 0.05, f'{direct:.2f}', 
                ha='center', va='bottom', fontsize=8)
        ax.text(i, convolved + 0.05, f'{convolved:.2f}',
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width, reference + 0.05, f'{reference:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {output_path}")
    plt.show()


def main():
    # Path to V3_outputs
    v3_outputs_dir = Path(__file__).parent / "V3_outputs"
    
    if not v3_outputs_dir.exists():
        print(f"❌ Directory not found: {v3_outputs_dir}")
        return
    
    # Find all subdirectories
    folders = [f for f in v3_outputs_dir.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"❌ No subdirectories found in {v3_outputs_dir}")
        return
    
    print(f"🔍 Found {len(folders)} folders to analyze\n")
    
    # Analyze each folder
    all_results = {}
    for folder in sorted(folders):
        metrics = analyze_folder(folder)
        if metrics is not None:
            all_results[folder.name] = metrics
    
    if not all_results:
        print("❌ No valid results found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_results, orient='index')
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df.describe())
    
    print("\n" + "="*80)
    print("FULL RESULTS")
    print("="*80)
    print(df.to_string())
    
    # Save to CSV
    csv_path = v3_outputs_dir / "lsd_analysis_results.csv"
    df.to_csv(csv_path)
    print(f"\n💾 Saved results to: {csv_path}")
    
    # Create visualization
    plot_path = v3_outputs_dir / "lsd_analysis_plot.png"
    create_visualization(df, plot_path)
    
    # Print insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Baseline: Target vs Reference
    reference_mean = df['target_vs_reference'].mean()
    print(f"📍 Baseline (Target vs Random Reference):")
    print(f"  - Mean LSD: {reference_mean:.4f} dB")
    
    # Best performing prediction method (vs target)
    direct_mean = df['target_vs_direct'].mean()
    convolved_mean = df['target_vs_convolved'].mean()
    better_method = "Direct" if direct_mean < convolved_mean else "Convolved"
    
    print(f"\n✓ Prediction Results (vs target):")
    print(f"  - Direct LSD mean: {direct_mean:.4f} dB")
    print(f"  - Convolved LSD mean: {convolved_mean:.4f} dB")
    print(f"  - Better method: {better_method}")
    
    # Improvement over baseline
    direct_improvement = reference_mean - direct_mean
    convolved_improvement = reference_mean - convolved_mean
    
    print(f"\n✅ Model Performance:")
    if direct_improvement > 0:
        print(f"  - Direct improved by: {direct_improvement:.4f} dB ({(direct_improvement/reference_mean*100):.1f}%)")
    else:
        print(f"  - Direct WORSE by: {abs(direct_improvement):.4f} dB ({(abs(direct_improvement)/reference_mean*100):.1f}%) ❌")
    
    if convolved_improvement > 0:
        print(f"  - Convolved improved by: {convolved_improvement:.4f} dB ({(convolved_improvement/reference_mean*100):.1f}%)")
    else:
        print(f"  - Convolved WORSE by: {abs(convolved_improvement):.4f} dB ({(abs(convolved_improvement)/reference_mean*100):.1f}%) ❌")
    
    # Most consistent method
    direct_std = df['target_vs_direct'].std()
    convolved_std = df['target_vs_convolved'].std()
    reference_std = df['target_vs_reference'].std()
    more_consistent = "Direct" if direct_std < convolved_std else "Convolved"
    
    print(f"\n✓ Consistency (std deviation):")
    print(f"  - Direct: {direct_std:.4f} dB")
    print(f"  - Convolved: {convolved_std:.4f} dB")
    print(f"  - Reference: {reference_std:.4f} dB")
    print(f"  - More consistent: {more_consistent}")
    
    # Best cases
    best_direct_case = df['target_vs_direct'].idxmin()
    best_convolved_case = df['target_vs_convolved'].idxmin()
    
    print(f"\n✓ Best predictions:")
    print(f"  - Direct: {best_direct_case} (LSD: {df.loc[best_direct_case, 'target_vs_direct']:.4f})")
    print(f"  - Convolved: {best_convolved_case} (LSD: {df.loc[best_convolved_case, 'target_vs_convolved']:.4f})")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
