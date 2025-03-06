import torch
import math

class DeepHoughFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, numangle, numrho):
        # input_feat: Tensor of shape (B, C, H, W)
        # numangle: int (number of angle bins)
        # numrho: int (number of distance/rho bins)
        B, C, H, W = input_feat.shape

        device = input_feat.device
        dtype = input_feat.dtype

        # Compute angle values (0 to π) and their sine/cosine
        angles = torch.linspace(0, math.pi, steps=numangle, endpoint=False, device=device, dtype=dtype)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # Coordinate grid for image [0..W-1] in x and [0..H-1] in y
        # Use center of image as origin for the Hough transform
        # X[i,j] = j (column index), Y[i,j] = i (row index)
        Y = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
        X = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
        # Convert to coordinates relative to center
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        X_flat = X.contiguous().view(-1)  # (H*W,)
        Y_flat = Y.contiguous().view(-1)  # (H*W,)
        x_rel = X_flat - cx
        y_rel = cy - Y_flat  # positive y_rel means upward from center

        # Maximum distance from center (diagonal radius)
        R_max = math.sqrt(float(cx**2 + cy**2))
        # Distance bin size
        if numrho > 1:
            dr = 2 * R_max / (numrho - 1)
        else:
            dr = 1.0  # degenerate case (not typical)

        # Compute for each image pixel its corresponding rho index at each angle
        # r = x_rel * cosθ + y_rel * sinθ
        # rho_index = round((r + R_max) / dr)
        # Shape of r_matrix: (H*W, numangle)
        r_matrix = x_rel.unsqueeze(1) * cos_theta + y_rel.unsqueeze(1) * sin_theta
        rho_indices = torch.round((r_matrix + R_max) / dr).to(torch.long)
        rho_indices.clamp_(0, numrho - 1)  # ensure within [0, numrho-1]

        # Compute linear indices in the flattened Hough space (angle,rho) -> single index
        # linear_index = angle_index * numrho + rho_index
        angle_idx = torch.arange(numangle, device=device, dtype=torch.long)
        # Broadcast angle_idx over all pixels and add rho indices
        param_indices = angle_idx.unsqueeze(0) * numrho + rho_indices  # shape: (H*W, numangle)
        param_indices_flat = param_indices.view(-1)  # flatten to length (H*W * numangle)

        # Prepare output accumulator tensor
        BxC = B * C
        param_count = numangle * numrho
        output_flat = torch.zeros((BxC, param_count), device=device, dtype=dtype)

        # Flatten input features to shape (B*C, H*W) and repeat each pixel value for each angle
        input_flat = input_feat.view(BxC, H * W)  # (B*C, N_pixels)
        # Repeat each column (pixel) value numangle times
        vals = input_flat.repeat_interleave(numangle, dim=1)  # (B*C, H*W * numangle)

        # Scatter-add: accumulate each pixel's value into the corresponding (angle,rho) bin
        # Use param_indices_flat for all samples
        index_2d = param_indices_flat.unsqueeze(0).expand(BxC, -1)  # (B*C, H*W * numangle)
        output_flat.scatter_add_(dim=1, index=index_2d, src=vals)
        # Reshape to (B, C, numangle, numrho)
        output = output_flat.view(B, C, numangle, numrho)

        # Save needed context for backward
        ctx.save_for_backward(param_indices_flat)
        ctx.H, ctx.W = H, W
        ctx.numangle = numangle
        ctx.numrho = numrho
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: shape (B, C, numangle, numrho)
        param_indices_flat, = ctx.saved_tensors
        H, W = ctx.H, ctx.W
        numangle = ctx.numangle
        numrho = ctx.numrho

        B, C = grad_output.shape[0], grad_output.shape[1]
        BxC = B * C
        # Flatten grad_output to (B*C, numangle * numrho)
        grad_out_flat = grad_output.contiguous().view(BxC, numangle * numrho)

        # Gather gradients for each pixel-angle contribution using the saved indices
        index_2d = param_indices_flat.unsqueeze(0).expand(BxC, -1)  # (B*C, H*W * numangle)
        grad_vals = grad_out_flat.gather(dim=1, index=index_2d)     # (B*C, H*W * numangle)
        # Reshape to (B*C, H*W, numangle)
        grad_vals = grad_vals.view(BxC, H * W, numangle)
        # Sum over angle dimension to get total grad for each pixel
        grad_pixels = grad_vals.sum(dim=2)  # (B*C, H*W)
        # Reshape back to (B, C, H, W)
        grad_input = grad_pixels.view(B, C, H, W)
        return grad_input, None, None

# Convenience function (same API as original)
def deep_hough_transform(input_feat, numangle, numrho):
    return DeepHoughFunction.apply(input_feat, numangle, numrho)
