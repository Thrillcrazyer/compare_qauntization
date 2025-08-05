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


def get_matrix(file_path='attn_map.npy'):
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
    plt.close()

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
    plt.close()

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

def weight_quant_general_8bit(w):
    """
    General 8-bit weight quantization with min-max scaling (not BitNet)
    Args:
        w: a weight tensor with shape [d, k]
    Returns:
        quantized, dequantized: quantized and dequantized weight tensors
    """
    min_val = w.min()
    max_val = w.max()
    scale = (max_val - min_val) / 255.0
    
    # Quantize to 8-bit unsigned integer
    quantized = torch.round((w - min_val) / scale).clamp(0, 255)
    quantized = quantized.to(torch.uint8)
    
    # Dequantize back to float
    dequantized = quantized.float() * scale + min_val
    
    return quantized, dequantized

def smooth_general_quantization_method(activations, weight_matrix, alpha=0.5):
    """
    SmoothQuant method with general 8-bit quantization (not BitNet)
    Args:
        activations: input activation tensor [batch, features]
        weight_matrix: input weight matrix [in_features, out_features]
        alpha: smoothing parameter
    Returns:
        quantized_activations, quantized_weights, dequantized_weights, smoothed_activations, smoothed_weights
    """
    # Apply smoothing first
    smoothed_activations, smoothed_weights = smooth_activations(activations, weight_matrix, alpha=alpha)
    
    # Apply general 8-bit quantization to smoothed activations
    quantized_activations = activation_quant(smoothed_activations)
    
    # Apply general 8-bit quantization to smoothed weights (min-max scaling)
    min_val = smoothed_weights.min()
    max_val = smoothed_weights.max()
    scale = (max_val - min_val) / 255.0
    
    # Quantize to 8-bit unsigned integer
    quantized_weights = torch.round((smoothed_weights - min_val) / scale).clamp(0, 255)
    quantized_weights = quantized_weights.to(torch.uint8)
    
    # Dequantize back to float
    dequantized_weights = quantized_weights.float() * scale + min_val
    
    return quantized_activations, quantized_weights, dequantized_weights, smoothed_activations, smoothed_weights

def llm_int8_quantization_method(activations, weight_matrix, outlier_threshold=6.0):
    """
    LLM.int8 quantization method with outlier detection and mixed precision
    Args:
        activations: input activation tensor [batch, features]
        weight_matrix: input weight matrix [in_features, out_features]
        outlier_threshold: threshold for outlier detection (default 6.0)
    Returns:
        quantized_activations, quantized_weights, dequantized_weights, outlier_mask
    """
    # Step 1: Detect outliers in activations
    # Outliers are features with values above the threshold
    outlier_mask = activations.abs().max(dim=0)[0] > outlier_threshold  # [features]
    
    # Step 2: Separate outlier and non-outlier features
    outlier_indices = torch.where(outlier_mask)[0]
    regular_indices = torch.where(~outlier_mask)[0]
    
    print(f"LLM.int8: Found {len(outlier_indices)} outlier features out of {len(outlier_mask)}")
    
    # Step 3: For regular features, apply 8-bit quantization
    if len(regular_indices) > 0:
        regular_activations = activations[:, regular_indices]
        regular_weights = weight_matrix[regular_indices, :]
        
        # Quantize regular activations (per-token quantization)
        act_scale = 127.0 / regular_activations.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        quantized_regular_act = (regular_activations * act_scale).round().clamp_(-128, 127)
        
        # Quantize regular weights (per-channel quantization)
        weight_scale = 127.0 / regular_weights.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)
        quantized_regular_weights = (regular_weights * weight_scale).round().clamp_(-128, 127)
        
        # Dequantize for computation
        dequant_regular_act = quantized_regular_act / act_scale
        dequant_regular_weights = quantized_regular_weights / weight_scale
    else:
        quantized_regular_act = torch.empty(0)
        quantized_regular_weights = torch.empty(0)
        dequant_regular_act = torch.empty(0)
        dequant_regular_weights = torch.empty(0)
    
    # Step 4: For outlier features, keep in FP16/FP32 (no quantization)
    if len(outlier_indices) > 0:
        outlier_activations = activations[:, outlier_indices]
        outlier_weights = weight_matrix[outlier_indices, :]
        # Keep outliers in original precision
        quantized_outlier_act = outlier_activations  # No quantization
        quantized_outlier_weights = outlier_weights   # No quantization
    else:
        quantized_outlier_act = torch.empty(0)
        quantized_outlier_weights = torch.empty(0)
    
    # Step 5: Reconstruct full tensors
    quantized_activations = torch.zeros_like(activations)
    quantized_weights = torch.zeros_like(weight_matrix)
    dequantized_weights = torch.zeros_like(weight_matrix)
    
    if len(regular_indices) > 0:
        quantized_activations[:, regular_indices] = dequant_regular_act
        quantized_weights[regular_indices, :] = quantized_regular_weights
        dequantized_weights[regular_indices, :] = dequant_regular_weights
    
    if len(outlier_indices) > 0:
        quantized_activations[:, outlier_indices] = quantized_outlier_act
        quantized_weights[outlier_indices, :] = quantized_outlier_weights
        dequantized_weights[outlier_indices, :] = quantized_outlier_weights  # No quantization loss
    
    return quantized_activations, quantized_weights, dequantized_weights, outlier_mask

def visualize_llm_int8_analysis(original_activations, original_weights, quantized_activations, 
                               quantized_weights, dequantized_weights, outlier_mask, method_name):
    """
    Specialized visualization for LLM.int8 showing outlier detection and mixed precision
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{method_name} - LLM.int8 Analysis with Outlier Detection', fontsize=16, fontweight='bold')
    
    # Convert to numpy
    orig_act_np = original_activations.cpu().numpy().flatten()
    orig_w_np = original_weights.cpu().numpy().flatten()
    quant_w_np = quantized_weights.cpu().numpy().flatten()
    dequant_w_np = dequantized_weights.cpu().numpy().flatten()
    outlier_mask_np = outlier_mask.cpu().numpy()
    
    # Top row: Activation analysis
    axes[0, 0].hist(orig_act_np, bins=100, alpha=0.7, color='blue', density=True, edgecolor='black')
    axes[0, 0].axvline(6.0, color='red', linestyle='--', linewidth=2, label='Outlier Threshold (6.0)')
    axes[0, 0].axvline(-6.0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Original Activation Distribution')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Outlier statistics
    num_outliers = outlier_mask_np.sum()
    outlier_ratio = num_outliers / len(outlier_mask_np) * 100
    
    axes[0, 1].bar(['Regular Features', 'Outlier Features'], 
                  [len(outlier_mask_np) - num_outliers, num_outliers],
                  color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_title(f'Feature Classification\n({outlier_ratio:.1f}% outliers)')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Show outlier pattern
    outlier_pattern = outlier_mask_np.astype(int)
    axes[0, 2].plot(outlier_pattern, color='red', linewidth=2)
    axes[0, 2].fill_between(range(len(outlier_pattern)), outlier_pattern, alpha=0.3, color='red')
    axes[0, 2].set_title('Outlier Feature Locations')
    axes[0, 2].set_xlabel('Feature Index')
    axes[0, 2].set_ylabel('Is Outlier (1/0)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Bottom row: Weight quantization analysis
    axes[1, 0].hist(orig_w_np, bins=100, alpha=0.7, color='blue', density=True, 
                   edgecolor='black', label='Original')
    axes[1, 0].hist(dequant_w_np, bins=100, alpha=0.5, color='green', density=True, 
                   edgecolor='black', label='Dequantized')
    axes[1, 0].set_title('Weight Distribution Comparison')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quantization error for regular vs outlier features
    regular_mask = ~outlier_mask_np
    if regular_mask.sum() > 0 and outlier_mask_np.sum() > 0:
        regular_error = np.abs(orig_w_np[:len(regular_mask)][regular_mask] - 
                              dequant_w_np[:len(regular_mask)][regular_mask])
        outlier_error = np.abs(orig_w_np[:len(outlier_mask_np)][outlier_mask_np] - 
                              dequant_w_np[:len(outlier_mask_np)][outlier_mask_np])
        
        axes[1, 1].hist(regular_error, bins=50, alpha=0.7, color='green', density=True,
                       label=f'Regular Features (MSE: {np.mean(regular_error**2):.6f})')
        axes[1, 1].hist(outlier_error, bins=50, alpha=0.7, color='red', density=True,
                       label=f'Outlier Features (MSE: {np.mean(outlier_error**2):.6f})')
        axes[1, 1].set_title('Quantization Error by Feature Type')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Overall statistics
    mse = np.mean((orig_w_np - dequant_w_np) ** 2)
    mae = np.mean(np.abs(orig_w_np - dequant_w_np))
    
    stats_text = f'Overall Statistics:\nMSE: {mse:.6f}\nMAE: {mae:.6f}\nOutlier Features: {num_outliers}\nOutlier Ratio: {outlier_ratio:.1f}%\nRegular Features Quantized: 8-bit\nOutlier Features: FP16/32'
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('LLM.int8 Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_llm_int8_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_weight_distribution_changes(original_weights, smoothed_weights, quantized_weights, dequantized_weights, method_name):
    """
    Visualize how weight distributions change through different processing steps
    Args:
        original_weights: Original weight tensor
        smoothed_weights: Smoothed weight tensor (if applicable)
        quantized_weights: Quantized weight tensor
        dequantized_weights: Dequantized weight tensor
        method_name: Name of the method for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{method_name} - Weight Distribution Evolution', fontsize=16, fontweight='bold')
    
    # Convert to numpy
    orig_np = original_weights.cpu().numpy().flatten()
    smooth_np = smoothed_weights.cpu().numpy().flatten() if smoothed_weights is not None else orig_np
    quant_np = quantized_weights.cpu().numpy().flatten()
    dequant_np = dequantized_weights.cpu().numpy().flatten()
    
    # Plot original distribution
    axes[0, 0].hist(orig_np, bins=100, alpha=0.7, color='blue', density=True, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Original Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(orig_np.mean(), color='red', linestyle='--', label=f'Mean: {orig_np.mean():.4f}')
    axes[0, 0].axvline(np.median(orig_np), color='orange', linestyle='--', label=f'Median: {np.median(orig_np):.4f}')
    axes[0, 0].legend()
    
    # Plot smoothed distribution (if different from original)
    if smoothed_weights is not None and not torch.equal(original_weights, smoothed_weights):
        axes[0, 1].hist(smooth_np, bins=100, alpha=0.7, color='green', density=True, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('Smoothed Weight Distribution')
        axes[0, 1].axvline(smooth_np.mean(), color='red', linestyle='--', label=f'Mean: {smooth_np.mean():.4f}')
        axes[0, 1].axvline(np.median(smooth_np), color='orange', linestyle='--', label=f'Median: {np.median(smooth_np):.4f}')
    else:
        axes[0, 1].hist(orig_np, bins=100, alpha=0.7, color='blue', density=True, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('Original Weight Distribution (No Smoothing)')
        axes[0, 1].axvline(orig_np.mean(), color='red', linestyle='--', label=f'Mean: {orig_np.mean():.4f}')
        axes[0, 1].axvline(np.median(orig_np), color='orange', linestyle='--', label=f'Median: {np.median(orig_np):.4f}')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot quantized distribution
    if quantized_weights.dtype == torch.uint8:
        unique_vals, counts = np.unique(quant_np, return_counts=True)
        axes[1, 0].bar(unique_vals, counts/len(quant_np), alpha=0.7, color='red', edgecolor='black', width=0.8)
        axes[1, 0].set_title('Quantized Weight Distribution (Discrete)')
        axes[1, 0].set_xlabel('Quantized Value (0-255)')
    else:
        axes[1, 0].hist(quant_np, bins=50, alpha=0.7, color='red', density=True, edgecolor='black', linewidth=0.5)
        axes[1, 0].set_title('Quantized Weight Distribution')
        axes[1, 0].set_xlabel('Quantized Value')
        axes[1, 0].axvline(quant_np.mean(), color='blue', linestyle='--', label=f'Mean: {quant_np.mean():.4f}')
        axes[1, 0].legend()
    axes[1, 0].set_ylabel('Density/Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot dequantized distribution
    axes[1, 1].hist(dequant_np, bins=100, alpha=0.7, color='purple', density=True, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('Dequantized Weight Distribution')
    axes[1, 1].set_xlabel('Weight Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(dequant_np.mean(), color='red', linestyle='--', label=f'Mean: {dequant_np.mean():.4f}')
    axes[1, 1].axvline(np.median(dequant_np), color='orange', linestyle='--', label=f'Median: {np.median(dequant_np):.4f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_weight_distribution_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_comparative_distributions(original_weights, methods_data, method_names):
    """
    Compare weight distributions across different quantization methods
    Args:
        original_weights: Original weight tensor
        methods_data: List of (quantized, dequantized) tuples for each method
        method_names: List of method names
    """
    fig, axes = plt.subplots(2, len(method_names) + 1, figsize=(4 * (len(method_names) + 1), 8))
    if len(method_names) == 1:
        axes = axes.reshape(2, -1)
    
    fig.suptitle('Comparative Weight Distribution Analysis', fontsize=16, fontweight='bold')
    
    orig_np = original_weights.cpu().numpy().flatten()
    
    # Plot original distribution
    axes[0, 0].hist(orig_np, bins=50, alpha=0.7, color='blue', density=True, edgecolor='black')
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].hist(orig_np, bins=50, alpha=0.7, color='blue', density=True, edgecolor='black')
    axes[1, 0].set_title('Original (Reference)')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    # Plot each method's results
    for i, ((quantized, dequantized), method_name) in enumerate(zip(methods_data, method_names)):
        col_idx = i + 1
        color = colors[i % len(colors)]
        
        quant_np = quantized.cpu().numpy().flatten()
        dequant_np = dequantized.cpu().numpy().flatten()
        
        # Top row: Quantized distributions
        if quantized.dtype == torch.uint8:
            unique_vals, counts = np.unique(quant_np, return_counts=True)
            axes[0, col_idx].bar(unique_vals, counts/len(quant_np), alpha=0.7, color=color, width=0.8)
            axes[0, col_idx].set_title(f'{method_name}\n(Quantized)')
        else:
            axes[0, col_idx].hist(quant_np, bins=50, alpha=0.7, color=color, density=True, edgecolor='black')
            axes[0, col_idx].set_title(f'{method_name}\n(Quantized)')
        axes[0, col_idx].grid(True, alpha=0.3)
        
        # Bottom row: Dequantized distributions
        axes[1, col_idx].hist(dequant_np, bins=50, alpha=0.7, color=color, density=True, edgecolor='black')
        axes[1, col_idx].set_title(f'{method_name}\n(Dequantized)')
        axes[1, col_idx].set_xlabel('Weight Value')
        axes[1, col_idx].grid(True, alpha=0.3)
        
        # Add overlay of original distribution for comparison
        axes[1, col_idx].hist(orig_np, bins=50, alpha=0.3, color='blue', density=True, 
                            edgecolor='blue', linestyle='--', histtype='step', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('comparative_weight_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_quantization_error_heatmap(original_weights, dequantized_weights, method_name):
    """
    Visualize quantization error as a heatmap for matrix weights
    Args:
        original_weights: Original weight matrix
        dequantized_weights: Dequantized weight matrix
        method_name: Name of the quantization method
    """
    # Calculate error
    error = (original_weights - dequantized_weights).cpu().numpy()
    
    # Take 42x42 portion if larger
    if error.shape[0] > 42 or error.shape[1] > 42:
        error = error[:42, :42]
        orig_portion = original_weights[:42, :42].cpu().numpy()
        dequant_portion = dequantized_weights[:42, :42].cpu().numpy()
    else:
        orig_portion = original_weights.cpu().numpy()
        dequant_portion = dequantized_weights.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{method_name} - Quantization Error Analysis (42x42)', fontsize=16, fontweight='bold')
    
    # Original weights heatmap
    im1 = axes[0].imshow(orig_portion, cmap='viridis', aspect='equal')
    axes[0].set_title('Original Weights')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Dequantized weights heatmap
    im2 = axes[1].imshow(dequant_portion, cmap='viridis', aspect='equal')
    axes[1].set_title('Dequantized Weights')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Error heatmap
    im3 = axes[2].imshow(error, cmap='RdBu_r', aspect='equal', vmin=-np.abs(error).max(), vmax=np.abs(error).max())
    axes[2].set_title('Quantization Error\n(Original - Dequantized)')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Add statistics
    mse = np.mean(error**2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    # Add text box with error statistics
    error_stats = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax |Error|: {max_error:.6f}'
    axes[2].text(0.02, 0.98, error_stats, transform=axes[2].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for ax in axes:
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Smoothing example with general 8-bit quantization
    print("\n--- SmoothQuant with General 8-bit Quantization Example ---")
    # Create sample activations and weights for smoothing
    sample_act = torch.randn(32, 256) * 2.0  # [batch, features]
    sample_w = torch.randn(256, 512) * 0.1   # [in_features, out_features]
    
    print(f"Original activation range: [{sample_act.min():.4f}, {sample_act.max():.4f}]")
    print(f"Original weight range: [{sample_w.min():.4f}, {sample_w.max():.4f}]")
    
    # Apply SmoothQuant with general 8-bit quantization
    smoothed_quant_act, smoothed_quant_w, smoothed_dequant_w, smoothed_act, smoothed_w = smooth_general_quantization_method(
        sample_act, sample_w, alpha=0.5
    )
    
    print(f"Smoothed activation range: [{smoothed_act.min():.4f}, {smoothed_act.max():.4f}]")
    print(f"Smoothed weight range: [{smoothed_w.min():.4f}, {smoothed_w.max():.4f}]")
    print(f"Quantized weight range: [{smoothed_quant_w.min()}, {smoothed_quant_w.max()}]")
    print(f"Dequantized weight range: [{smoothed_dequant_w.min():.4f}, {smoothed_dequant_w.max():.4f}]")
    
    # Visualize SmoothQuant + 8-bit quantization results
    visualize_quantization_results(smoothed_w, smoothed_quant_w, smoothed_dequant_w, "SmoothQuant + 8-bit Quantization")
    
    # Create a 42x42 sample for smoothed matrix visualization
    sample_act_42x42 = torch.randn(32, 42) * 2.0
    sample_w_42x42 = torch.randn(42, 42) * 0.1
    smoothed_quant_act_42x42, smoothed_quant_w_42x42, smoothed_dequant_w_42x42, smoothed_act_42x42_out, smoothed_w_42x42_out = smooth_general_quantization_method(
        sample_act_42x42, sample_w_42x42, alpha=0.5
    )
    visualize_matrix_heatmaps(smoothed_w_42x42_out, smoothed_quant_w_42x42, smoothed_dequant_w_42x42, "SmoothQuant + 8-bit Quantization")
    
    # Compare different quantization methods
    print("\n--- Quantization Method Comparison ---")
    
    # Without smoothing - general 8-bit
    quant_w_direct, dequant_w_direct = weight_quant_general_8bit(sample_w)
    mse_direct = torch.mean((sample_w - dequant_w_direct) ** 2)
    
    # With smoothing - general 8-bit  
    mse_smoothed = torch.mean((smoothed_w - smoothed_dequant_w) ** 2)
    
    # Activation comparison
    quant_act_direct = activation_quant(sample_act)
    mse_act_direct = torch.mean((sample_act - quant_act_direct) ** 2)
    mse_act_smoothed = torch.mean((smoothed_act - smoothed_quant_act) ** 2)
    
    print(f"Weight MSE without smoothing (8-bit): {mse_direct:.6f}")
    print(f"Weight MSE with SmoothQuant (8-bit): {mse_smoothed:.6f}")
    print(f"Weight smoothing improvement: {((mse_direct - mse_smoothed) / mse_direct * 100):.2f}%")
    
    print(f"Activation MSE without smoothing: {mse_act_direct:.6f}")
    print(f"Activation MSE with SmoothQuant: {mse_act_smoothed:.6f}")
    print(f"Activation smoothing improvement: {((mse_act_direct - mse_act_smoothed) / mse_act_direct * 100):.2f}%")
    
    # LLM.int8 quantization example
    print("\n--- LLM.int8 Quantization Example ---")
    # Create sample with some outlier features
    sample_act_llm = torch.randn(32, 256) * 2.0
    # Inject some outliers (features with large values)
    outlier_indices = torch.randperm(256)[:20]  # 20 outlier features
    sample_act_llm[:, outlier_indices] *= 5.0  # Make them outliers
    
    sample_w_llm = torch.randn(256, 512) * 0.1
    
    print(f"LLM.int8 activation range: [{sample_act_llm.min():.4f}, {sample_act_llm.max():.4f}]")
    print(f"LLM.int8 weight range: [{sample_w_llm.min():.4f}, {sample_w_llm.max():.4f}]")
    
    # Apply LLM.int8 quantization
    llm_quant_act, llm_quant_w, llm_dequant_w, llm_outlier_mask = llm_int8_quantization_method(
        sample_act_llm, sample_w_llm, outlier_threshold=6.0
    )
    
    # Visualize LLM.int8 results
    visualize_llm_int8_analysis(sample_act_llm, sample_w_llm, llm_quant_act, 
                               llm_quant_w, llm_dequant_w, llm_outlier_mask, "LLM.int8 Quantization")
    
    # Create 42x42 sample for LLM.int8 matrix visualization
    sample_act_llm_42x42 = torch.randn(32, 42) * 2.0
    sample_act_llm_42x42[:, :5] *= 5.0  # Make first 5 features outliers
    sample_w_llm_42x42 = torch.randn(42, 42) * 0.1
    
    llm_quant_act_42x42, llm_quant_w_42x42, llm_dequant_w_42x42, llm_outlier_mask_42x42 = llm_int8_quantization_method(
        sample_act_llm_42x42, sample_w_llm_42x42, outlier_threshold=6.0
    )
    
    visualize_matrix_heatmaps(sample_w_llm_42x42, llm_quant_w_42x42, llm_dequant_w_42x42, "LLM.int8 Quantization")
    
    # Additional visualizations for weight distribution changes
    print("\n--- Advanced Weight Distribution Analysis ---")
    
    # 1. Visualize weight distribution evolution for SmoothQuant + 8-bit
    visualize_weight_distribution_changes(sample_w, smoothed_w, smoothed_quant_w, smoothed_dequant_w, 
                                        "SmoothQuant + 8-bit Quantization")
    
    # 2. Visualize weight distribution evolution for general 8-bit (no smoothing)
    visualize_weight_distribution_changes(sample_w, None, quant_w_direct, dequant_w_direct, 
                                        "General 8-bit Quantization")
    
    # 3. Visualize weight distribution evolution for 1.58-bit quantization
    visualize_weight_distribution_changes(sample_weight, None, quantized_weight_158, dequantized_weight_158, 
                                        "1.58-bit Ternary Quantization")
    
    # 3.5. Visualize weight distribution evolution for LLM.int8 quantization
    visualize_weight_distribution_changes(sample_w_llm, None, llm_quant_w, llm_dequant_w, 
                                        "LLM.int8 Quantization")
    
    # 4. Comparative distribution analysis
    methods_data = [
        (quant_w_direct, dequant_w_direct),  # General 8-bit
        (smoothed_quant_w, smoothed_dequant_w),  # SmoothQuant + 8-bit
        (quantized_weight_158, dequantized_weight_158),  # 1.58-bit
        (llm_quant_w, llm_dequant_w)  # LLM.int8
    ]
    method_names = ["General 8-bit", "SmoothQuant + 8-bit", "1.58-bit Ternary", "LLM.int8"]
    
    visualize_comparative_distributions(sample_w, methods_data, method_names)
    
    # 5. Error heatmaps for different methods
    visualize_quantization_error_heatmap(sample_w_42x42, dequant_w_direct[:42, :42], "General 8-bit Quantization")
    visualize_quantization_error_heatmap(sample_w_42x42, smoothed_dequant_w_42x42, "SmoothQuant + 8-bit Quantization")
    visualize_quantization_error_heatmap(sample_weight_42x42, dequantized_42x42, "1.58-bit Ternary Quantization")
    visualize_quantization_error_heatmap(sample_w_llm_42x42, llm_dequant_w_42x42, "LLM.int8 Quantization")
    
    print("\n--- Distribution Statistics Summary ---")
    
    # Calculate and display comprehensive statistics
    def calculate_distribution_stats(original, quantized, dequantized, method_name):
        orig_np = original.cpu().numpy().flatten()
        dequant_np = dequantized.cpu().numpy().flatten()
        
        mse = np.mean((orig_np - dequant_np) ** 2)
        mae = np.mean(np.abs(orig_np - dequant_np))
        snr = 10 * np.log10(np.var(orig_np) / mse) if mse > 0 else float('inf')
        
        orig_std = np.std(orig_np)
        dequant_std = np.std(dequant_np)
        std_ratio = dequant_std / orig_std if orig_std > 0 else 0
        
        if quantized.dtype == torch.uint8:
            unique_values = len(torch.unique(quantized))
        else:
            unique_values = len(torch.unique(quantized.round()))
        
        print(f"\n{method_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  Std ratio (dequant/orig): {std_ratio:.4f}")
        print(f"  Unique quantized values: {unique_values}")
        print(f"  Original range: [{orig_np.min():.4f}, {orig_np.max():.4f}]")
        print(f"  Dequantized range: [{dequant_np.min():.4f}, {dequant_np.max():.4f}]")
    
    calculate_distribution_stats(sample_w, quant_w_direct, dequant_w_direct, "General 8-bit Quantization")
    calculate_distribution_stats(sample_w, smoothed_quant_w, smoothed_dequant_w, "SmoothQuant + 8-bit Quantization")
    calculate_distribution_stats(sample_weight, quantized_weight_158, dequantized_weight_158, "1.58-bit Ternary Quantization")
    calculate_distribution_stats(sample_w_llm, llm_quant_w, llm_dequant_w, "LLM.int8 Quantization")
    
    print(f"\n--- LLM.int8 Special Analysis ---")
    outlier_count = llm_outlier_mask.sum().item()
    total_features = len(llm_outlier_mask)
    outlier_ratio = (outlier_count / total_features) * 100
    print(f"Outlier features detected: {outlier_count}/{total_features} ({outlier_ratio:.1f}%)")
    print(f"Regular features (8-bit quantized): {total_features - outlier_count}")
    print(f"Outlier features (FP16/32 precision): {outlier_count}")
    
    # Memory savings calculation
    regular_bits = (total_features - outlier_count) * 8  # 8-bit quantized
    outlier_bits = outlier_count * 16  # Assume FP16 for outliers
    original_bits = total_features * 32  # Original FP32
    memory_savings = ((original_bits - (regular_bits + outlier_bits)) / original_bits) * 100
    print(f"Estimated memory savings: {memory_savings:.1f}%")

