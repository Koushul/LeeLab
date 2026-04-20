import numpy as np
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def estimate_noise(signal):
    """Estimate noise floor using IQR-based robust statistics."""
    q1, median, q3 = np.percentile(signal, [25, 50, 75])
    iqr = q3 - q1
    return median, iqr


def find_best_barcode(signal, barcode_names, threshold):
    """
    Find the single most likely true barcode in a signal segment.
    
    Steps:
        1. Median filter to smooth noise
        2. Subtract noise floor
        3. Find peaks above adaptive threshold
        4. Compute signal-weighted centroid of peak cluster
        5. Select highest peak near centroid
    """
    if len(signal) == 0:
        return None, 0.0, {}
    
    # Median filter (window=3)
    filtered = medfilt(signal, kernel_size=3)
    
    # Subtract noise floor
    cleaned = np.maximum(filtered - threshold * 0.3, 0)
    
    # Find peaks above threshold
    peak_mask = cleaned > threshold * 0.5
    peak_indices = np.where(peak_mask)[0]
    
    if len(peak_indices) == 0:
        # Fallback: pick the max of raw signal
        best_idx = np.argmax(signal)
        return barcode_names[best_idx], 0.0, {
            'method': 'max_fallback',
            'peak_value': signal[best_idx],
            'n_peaks': 0,
        }
    
    peak_signals = cleaned[peak_indices]
    
    # Signal-weighted centroid
    centroid = np.average(peak_indices, weights=peak_signals)
    
    # Highest peak
    max_peak_local = np.argmax(peak_signals)
    max_peak_idx = peak_indices[max_peak_local]
    
    # Choose: highest peak if close to centroid, else nearest to centroid
    if abs(max_peak_idx - centroid) <= 3:
        chosen_idx = max_peak_idx
    else:
        distances = np.abs(peak_indices - centroid)
        chosen_idx = peak_indices[np.argmin(distances)]
    
    # Confidence: fraction of total signal in chosen peak
    total_signal = cleaned.sum()
    confidence = cleaned[chosen_idx] / total_signal if total_signal > 0 else 0.0
    
    return barcode_names[chosen_idx], confidence, {
        'method': 'weighted_peak',
        'peak_value': signal[chosen_idx],
        'n_peaks': len(peak_indices),
    }


def demux_cell(signal_half1, signal_half2, names_half1, names_half2, threshold_k=2.5):
    """
    Demultiplex a single cell given its two barcode halves.
    """
    # Compute threshold from the full signal (both halves combined)
    full_signal = np.concatenate([signal_half1, signal_half2])
    median, iqr = estimate_noise(full_signal)
    threshold = median + threshold_k * iqr
    
    bc1, conf1, info1 = find_best_barcode(signal_half1, names_half1, threshold)
    bc2, conf2, info2 = find_best_barcode(signal_half2, names_half2, threshold)
    
    return {
        'x': bc1,
        'y': bc2,
        'confidence_1': round(conf1, 4),
        'confidence_2': round(conf2, 4),
        'threshold': round(threshold, 3),
        'noise_median': round(median, 3),
        'noise_iqr': round(iqr, 3),
        'n_peaks_1': info1.get('n_peaks', 0),
        'n_peaks_2': info2.get('n_peaks', 0),
        'method_1': info1.get('method', ''),
        'method_2': info2.get('method', ''),
    }


def demux_all_cells(prot_DF_filtered, threshold_k=2.5):
    """
    Demultiplex all cells.
    
    Parameters:
        prot_DF_filtered: DataFrame, rows = cells, columns = spatial barcodes
                          Columns named sbc1..sbc48 (half 1) and sbc96..sbc143 (half 2)
        threshold_k: IQR multiplier for noise threshold (default 2.5)
    
    Returns:
        DataFrame with one row per cell: barcode_1, barcode_2, confidences, diagnostics
    """
    all_cols = prot_DF_filtered.columns.tolist()
    
    # Split into two halves based on barcode numbering
    col_nums = {}
    for c in all_cols:
        num = int(''.join(filter(str.isdigit, c)))
        col_nums[c] = num
    
    sorted_cols = sorted(all_cols, key=lambda c: col_nums[c])
    nums = [col_nums[c] for c in sorted_cols]
    
    # Find the natural split (gap between 48 and 96)
    gaps = [(nums[i+1] - nums[i], i) for i in range(len(nums)-1)]
    biggest_gap_idx = max(gaps, key=lambda x: x[0])[1]
    
    half1_cols = sorted_cols[:biggest_gap_idx + 1]
    half2_cols = sorted_cols[biggest_gap_idx + 1:]
    
    print(f"Half 1: {len(half1_cols)} barcodes ({half1_cols[0]} -> {half1_cols[-1]})")
    print(f"Half 2: {len(half2_cols)} barcodes ({half2_cols[0]} -> {half2_cols[-1]})")
    print(f"Total cells: {len(prot_DF_filtered)}")
    print(f"Threshold multiplier: {threshold_k}")
    print("Running demux...")
    
    names_h1 = np.array(half1_cols)
    names_h2 = np.array(half2_cols)
    
    results = []
    for cell_id in prot_DF_filtered.index:
        sig_h1 = prot_DF_filtered.loc[cell_id, half1_cols].values.astype(float)
        sig_h2 = prot_DF_filtered.loc[cell_id, half2_cols].values.astype(float)
        
        res = demux_cell(sig_h1, sig_h2, names_h1, names_h2, threshold_k=threshold_k)
        res['cell'] = cell_id
        results.append(res)
    
    result_df = pd.DataFrame(results)
    cols = ['cell', 'x', 'y', 'confidence_1', 'confidence_2',
            'threshold', 'noise_median', 'noise_iqr',
            'n_peaks_1', 'n_peaks_2', 'method_1', 'method_2']
    result_df = result_df[cols]
    
    # Summary stats
    low1 = (result_df['confidence_1'] < 0.1).sum()
    low2 = (result_df['confidence_2'] < 0.1).sum()
    fallback1 = (result_df['method_1'] == 'max_fallback').sum()
    fallback2 = (result_df['method_2'] == 'max_fallback').sum()
    print(f"\nDone! Results summary:")
    print(f"  Low confidence (<10%) -- half1: {low1}, half2: {low2}")
    print(f"  Fallback (no peaks) -- half1: {fallback1}, half2: {fallback2}")
    
    return result_df


def plot_cell(prot_DF_filtered, cell_id, threshold_k=2.5, ax=None):
    """Plot the signal and detected barcodes for a single cell."""
    all_cols = prot_DF_filtered.columns.tolist()
    col_nums = {c: int(''.join(filter(str.isdigit, c))) for c in all_cols}
    sorted_cols = sorted(all_cols, key=lambda c: col_nums[c])
    
    signal = prot_DF_filtered.loc[cell_id, sorted_cols].values.astype(float)
    barcode_names = np.array(sorted_cols)
    
    # Find split
    nums = [col_nums[c] for c in sorted_cols]
    gaps = [(nums[i+1] - nums[i], i) for i in range(len(nums)-1)]
    split_idx = max(gaps, key=lambda x: x[0])[1] + 1
    
    # Run demux
    names_h1 = barcode_names[:split_idx]
    names_h2 = barcode_names[split_idx:]
    sig_h1 = signal[:split_idx]
    sig_h2 = signal[split_idx:]
    
    full_signal = np.concatenate([sig_h1, sig_h2])
    median, iqr = estimate_noise(full_signal)
    threshold = median + threshold_k * iqr
    
    bc1, conf1, _ = find_best_barcode(sig_h1, names_h1, threshold)
    bc2, conf2, _ = find_best_barcode(sig_h2, names_h2, threshold)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    
    x = np.arange(len(signal))
    filtered = medfilt(signal, kernel_size=3)
    
    ax.fill_between(x, signal, alpha=0.15, color='steelblue')
    ax.plot(x, signal, color='steelblue', alpha=0.4, linewidth=0.8, label='Raw')
    ax.plot(x, filtered, color='dodgerblue', linewidth=1.5, label='Median filtered')
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.6, linewidth=0.8,
               label=f'Threshold ({threshold:.1f})')
    ax.axvline(split_idx, color='orange', linestyle=':', alpha=0.5, label='Half split')
    
    # Mark best barcodes
    for bc, conf in [(bc1, conf1), (bc2, conf2)]:
        if bc is not None:
            idx = np.where(barcode_names == bc)[0]
            if len(idx) > 0:
                idx = idx[0]
                ax.axvline(idx, color='limegreen', linewidth=2, alpha=0.8)
                ax.annotate(f'* {bc}\n({conf:.1%})',
                            xy=(idx, signal[idx]), xytext=(idx + 1.5, signal[idx] + 1),
                            fontsize=8, color='green', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    ax.set_xticks(x[::4])
    ax.set_xticklabels([barcode_names[i] for i in x[::4]], rotation=45, fontsize=6)
    ax.set_title(f'Cell: {cell_id}', fontsize=10, fontweight='bold')
    ax.set_ylabel('UMI Count')
    ax.legend(fontsize=7, loc='upper right')
    
    return ax


def plot_sample_cells(prot_DF_filtered, n=6, threshold_k=2.5):
    """Plot a grid of sample cells for QC."""
    cells = prot_DF_filtered.index.tolist()
    sample = np.random.choice(cells, size=min(n, len(cells)), replace=False)
    
    nrows = (len(sample) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    
    for i, cell_id in enumerate(sample):
        plot_cell(prot_DF_filtered, cell_id, threshold_k=threshold_k, ax=axes[i])
    
    for j in range(len(sample), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('demux_sample_cells.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# USAGE
# ============================================================
# prot_DF_filtered = pd.read_csv('raw_BCs.csv', index_col=0)
# result_df = demux_all_cells(prot_DF_filtered, threshold_k=2.5)
# print(result_df.head(10))
#
# # QC plots
# plot_sample_cells(prot_DF_filtered, n=6)
#
# # Single cell plot
# plot_cell(prot_DF_filtered, 'AAACCCAAGGTCCCTG-1')
#
# # Filter low-confidence
# high_conf = result_df[(result_df['confidence_1'] > 0.05) & (result_df['confidence_2'] > 0.05)]
