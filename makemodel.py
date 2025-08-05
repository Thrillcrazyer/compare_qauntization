import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set Times New Roman font
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def get_rotation_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_matrix(file_path='final_avg_attn.npy'):
    matrix=np.load(file_path, allow_pickle=True)
    if isinstance(matrix, np.ndarray):
        matrix = torch.tensor(matrix, dtype=torch.float64)
    else:
        raise ValueError("The loaded matrix is not a valid numpy array.")
    return matrix.to("cuda" if torch.cuda.is_available() else "cpu")

def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
        x: an activation tensor with shape [n, d]
    Returns:
        y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant_with_dequant(w):
    """
    Per-tensor quantization to 1.58 bits with dequantization
    Args:
        w: a weight tensor with shape [d, k]
    Returns:
        quantized, dequantized: quantized and dequantized weight tensors
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    quantized = (w * scale).round().clamp_(-1, 1)
    dequantized = quantized / scale
    return quantized, dequantized

def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """
    Smooth activations and weights using SmoothQuant method
    Args:
        ln: LayerNorm layer
        fcs: List of linear layers or single linear layer
        act_scales: Activation scales tensor
        alpha: Smoothing parameter (0.5 default)
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # Calculate weight scales
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    
    # Calculate smoothing scales
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    # Apply smoothing to LayerNorm
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)
    
    # Apply smoothing to linear layers
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

def smooth_activations(activations, weights, alpha=0.5):
    """
    Simple smoothing function for activations and weights
    Args:
        activations: Input activation tensor [batch, in_features]
        weights: Weight tensor [in_features, out_features]
        alpha: Smoothing parameter
    Returns:
        smoothed_activations, smoothed_weights
    """
    # Calculate scales based on input features (dimension that matches)
    act_scales = activations.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)  # [1, in_features]
    weight_scales = weights.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)   # [in_features, 1]
    
    # Make sure dimensions match for broadcasting
    act_scales = act_scales.squeeze(0)      # [in_features]
    weight_scales = weight_scales.squeeze(1) # [in_features]
    
    # Smoothing scales
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    
    # Apply smoothing
    smoothed_activations = activations / scales.unsqueeze(0)  # Broadcast over batch dimension
    smoothed_weights = weights * scales.unsqueeze(1)          # Broadcast over output dimension
    
    return smoothed_activations, smoothed_weights

def visualize_quantization_results(original_weights, quantized_weights, dequantized_weights, method_name):
    """
    Visualize quantization results comparing original, quantized, and dequantized weights
    Args:
        original_weights: Original weight tensor
        quantized_weights: Quantized weight tensor (uint8)
        dequantized_weights: Dequantized weight tensor (float)
        method_name: Name of the quantization method
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{method_name} - Weight Quantization Analysis', fontsize=16, fontweight='bold')
    
    # Convert to numpy for visualization
    orig_np = original_weights.cpu().numpy().flatten()
    quant_np = quantized_weights.cpu().numpy().flatten() if quantized_weights.dtype == torch.uint8 else quantized_weights.cpu().numpy().flatten()
    dequant_np = dequantized_weights.cpu().numpy().flatten()
    
    # First row: Histograms
    axes[0, 0].hist(orig_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Original Weights Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(quant_np, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('Quantized Weights Distribution')
    axes[0, 1].set_xlabel('Quantized Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(dequant_np, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_title('Dequantized Weights Distribution')
    axes[0, 2].set_xlabel('Weight Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Second row: Comparison plots
    sample_indices = np.random.choice(len(orig_np), min(1000, len(orig_np)), replace=False)
    sample_orig = orig_np[sample_indices]
    sample_dequant = dequant_np[sample_indices]
    
    axes[1, 0].scatter(sample_orig, sample_dequant, alpha=0.6, s=1)
    axes[1, 0].plot([sample_orig.min(), sample_orig.max()], [sample_orig.min(), sample_orig.max()], 'r--', lw=2)
    axes[1, 0].set_title('Original vs Dequantized')
    axes[1, 0].set_xlabel('Original Weight')
    axes[1, 0].set_ylabel('Dequantized Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    error = orig_np - dequant_np
    axes[1, 1].hist(error, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Quantization Error Distribution')
    axes[1, 1].set_xlabel('Error (Original - Dequantized)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics
    mse = np.mean(error**2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    stats_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax Error: {max_error:.6f}\nOriginal Range: [{orig_np.min():.4f}, {orig_np.max():.4f}]\nDequantized Range: [{dequant_np.min():.4f}, {dequant_np.max():.4f}]'
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].set_title('Quantization Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_quantization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_matrix_heatmaps(original_matrix, quantized_matrix, dequantized_matrix, method_name):
    """
    Visualize 42x42 matrices as heatmaps with values displayed
    Args:
        original_matrix: Original matrix (42x42)
        quantized_matrix: Quantized matrix (42x42) 
        dequantized_matrix: Dequantized matrix (42x42)
        method_name: Name of the quantization method
    """
    # Take only the first 42x42 portion if larger
    if original_matrix.shape[0] > 42 or original_matrix.shape[1] > 42:
        original_matrix = original_matrix[:42, :42]
        quantized_matrix = quantized_matrix[:42, :42]
        dequantized_matrix = dequantized_matrix[:42, :42]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'{method_name} - 42x42 Matrix Heatmaps', fontsize=16, fontweight='bold')
    
    # Convert to numpy
    orig_np = original_matrix.cpu().numpy()
    quant_np = quantized_matrix.cpu().numpy()
    dequant_np = dequantized_matrix.cpu().numpy()
    
    # Original matrix heatmap
    im1 = axes[0].imshow(orig_np, cmap='viridis', aspect='equal')
    axes[0].set_title('Original Matrix')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Add text annotations for quantized values (only show every 4th value to avoid clutter)
    for i in range(0, 42, 4):
        for j in range(0, 42, 4):
            axes[0].text(j, i, f'{orig_np[i, j]:.3f}', 
                        ha='center', va='center', fontsize=6, color='white')
    
    # Quantized matrix heatmap
    im2 = axes[1].imshow(quant_np, cmap='plasma', aspect='equal')
    axes[1].set_title('Quantized Matrix')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Add text annotations for quantized values
    for i in range(0, 42, 4):
        for j in range(0, 42, 4):
            value = int(quant_np[i, j]) if quant_np.dtype in [np.uint8, np.int8] else quant_np[i, j]
            axes[1].text(j, i, f'{value}', 
                        ha='center', va='center', fontsize=6, color='white')
    
    # Dequantized matrix heatmap
    im3 = axes[2].imshow(dequant_np, cmap='coolwarm', aspect='equal')
    axes[2].set_title('Dequantized Matrix')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Add text annotations for dequantized values
    for i in range(0, 42, 4):
        for j in range(0, 42, 4):
            axes[2].text(j, i, f'{dequant_np[i, j]:.3f}', 
                        ha='center', va='center', fontsize=6, color='white')
    
    # Set labels and ticks
    for ax in axes:
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_xticks(range(0, 42, 5))
        ax.set_yticks(range(0, 42, 5))
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_matrix_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def general_quantization_method(activations, weight_matrix):
    """
    General quantization method with min-max scaling to 8-bit
    Args:
        activations: input activation tensor
        weight_matrix: input weight matrix
    Returns:
        quantized_activations, quantized_weights, dequantized_weights
    """
    # Quantize activations using activation_quant
    quantized_activations = activation_quant(activations) if activations is not None else None
    
    # Quantize weights - Min-max scaling to 8-bit
    min_val = weight_matrix.min()
    max_val = weight_matrix.max()
    scale = (max_val - min_val) / 255.0
    
    # Quantize to 8-bit unsigned integer
    quantized_weights = torch.round((weight_matrix - min_val) / scale).clamp(0, 255)
    quantized_weights = quantized_weights.to(torch.uint8)  # Ensure it's in uint8 format
    
    # Dequantize back to float for further processing
    dequantized_weights = quantized_weights.float() * scale + min_val
    
    return quantized_activations, quantized_weights, dequantized_weights



if __name__ == "__main__":
    # Load weight matrix
    Weight = get_matrix()
    print(f"Original weight shape: {Weight.shape}")
    print(f"Original weight min: {Weight.min():.6f}, max: {Weight.max():.6f}")
    
    # Generate rotation matrix and apply rotation
    size = 42  # Example size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rotation_matrix = get_rotation_matrix(size, device)
    rotated_weight = torch.matmul(Weight, rotation_matrix)
    print(f"Rotated weight shape: {rotated_weight.shape}")
    
    # Apply general quantization
    sample_activation_for_general = torch.randn(10, rotated_weight.shape[0]) * 2.0
    quantized_act, quantized_weight, dequantized_weight = general_quantization_method(sample_activation_for_general, rotated_weight)
    print(f"Quantized weight shape: {quantized_weight.shape}")
    print(f"Dequantized weight min: {dequantized_weight.min():.6f}, max: {dequantized_weight.max():.6f}")
    
    # Visualize general quantization results
    visualize_quantization_results(rotated_weight, quantized_weight, dequantized_weight, "General 8-bit Quantization")
    
    # Visualize 42x42 matrix heatmaps for general quantization
    visualize_matrix_heatmaps(rotated_weight, quantized_weight, dequantized_weight, "General 8-bit Quantization")
    
    # Example for activation quantization
    print("\n--- Activation Quantization Example ---")
    # Create a sample activation tensor [batch_size, feature_dim]
    sample_activation = torch.randn(10, 128) * 5.0  # Random activation with larger range
    print(f"Original activation shape: {sample_activation.shape}")
    print(f"Original activation range: [{sample_activation.min():.4f}, {sample_activation.max():.4f}]")
    
    quantized_activation = activation_quant(sample_activation)
    print(f"Quantized activation range: [{quantized_activation.min():.4f}, {quantized_activation.max():.4f}]")
    
    # Example for weight quantization (1.58-bit)
    print("\n--- Weight Quantization (1.58-bit) Example ---")
    sample_weight = torch.randn(256, 128) * 0.1  # Random weight matrix
    print(f"Original weight shape: {sample_weight.shape}")
    print(f"Original weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    
    quantized_weight_158, dequantized_weight_158 = weight_quant_with_dequant(sample_weight)
    print(f"1.58-bit quantized weight range: [{quantized_weight_158.min():.4f}, {quantized_weight_158.max():.4f}]")
    print(f"Unique values in quantized weight: {torch.unique(quantized_weight_158).numel()}")
    
    # Visualize 1.58-bit quantization results
    visualize_quantization_results(sample_weight, quantized_weight_158, dequantized_weight_158, "1.58-bit Ternary Quantization")
    
    # Create a 42x42 sample for matrix visualization
    sample_weight_42x42 = torch.randn(42, 42) * 0.1
    quantized_42x42, dequantized_42x42 = weight_quant_with_dequant(sample_weight_42x42)
    visualize_matrix_heatmaps(sample_weight_42x42, quantized_42x42, dequantized_42x42, "1.58-bit Ternary Quantization")
    
    # Compare quantization errors
    print("\n--- Quantization Error Analysis ---")
    mse_general = torch.mean((rotated_weight - dequantized_weight) ** 2)
    mse_weight = torch.mean((sample_weight - dequantized_weight_158) ** 2)
    mse_activation = torch.mean((sample_activation - quantized_activation) ** 2)
    
    print(f"General quantization MSE: {mse_general:.6f}")
    print(f"1.58-bit weight quantization MSE: {mse_weight:.6f}")
    print(f"Activation quantization MSE: {mse_activation:.6f}")
    
    # Smoothing example
    print("\n--- Smoothing Example ---")
    # Create sample activations and weights for smoothing
    sample_act = torch.randn(32, 256) * 2.0  # [batch, features]
    sample_w = torch.randn(256, 512) * 0.1   # [in_features, out_features]
    
    print(f"Original activation range: [{sample_act.min():.4f}, {sample_act.max():.4f}]")
    print(f"Original weight range: [{sample_w.min():.4f}, {sample_w.max():.4f}]")
    
    # Apply smoothing
    smoothed_act, smoothed_w = smooth_activations(sample_act, sample_w, alpha=0.5)
    
    print(f"Smoothed activation range: [{smoothed_act.min():.4f}, {smoothed_act.max():.4f}]")
    print(f"Smoothed weight range: [{smoothed_w.min():.4f}, {smoothed_w.max():.4f}]")
    
    # Apply quantization after smoothing
    smoothed_quant_act = activation_quant(smoothed_act)
    smoothed_quant_w, smoothed_dequant_w = weight_quant_with_dequant(smoothed_w)
    
    # Visualize smoothing quantization results
    visualize_quantization_results(smoothed_w, smoothed_quant_w, smoothed_dequant_w, "Smoothed + 1.58-bit Quantization")
    
    # Create a 42x42 sample for smoothed matrix visualization
    sample_act_42x42 = torch.randn(32, 42) * 2.0
    sample_w_42x42 = torch.randn(42, 42) * 0.1
    smoothed_act_42x42, smoothed_w_42x42 = smooth_activations(sample_act_42x42, sample_w_42x42, alpha=0.5)
    smoothed_quant_42x42, smoothed_dequant_42x42 = weight_quant_with_dequant(smoothed_w_42x42)
    visualize_matrix_heatmaps(smoothed_w_42x42, smoothed_quant_42x42, smoothed_dequant_42x42, "Smoothed + 1.58-bit Quantization")
    
    # Compare MSE with and without smoothing
    mse_original = torch.mean((sample_act - activation_quant(sample_act)) ** 2)
    mse_smoothed = torch.mean((smoothed_act - smoothed_quant_act) ** 2)
    
    print(f"Activation MSE without smoothing: {mse_original:.6f}")
    print(f"Activation MSE with smoothing: {mse_smoothed:.6f}")
    print(f"Smoothing improvement: {((mse_original - mse_smoothed) / mse_original * 100):.2f}%")

