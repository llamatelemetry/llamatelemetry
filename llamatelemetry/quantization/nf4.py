"""
NF4 (4-bit NormalFloat) Quantization

Implements NF4 quantization compatible with bitsandbytes and Unsloth.
NF4 is optimized for normally distributed weights and provides better quality
than uniform quantization at 4-bits.

References:
    - QLoRA paper: https://arxiv.org/abs/2305.14314
    - bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


# NF4 quantization lookup table (from QLoRA paper)
NF4_QUANT_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float32)


@dataclass
class NF4Config:
    """Configuration for NF4 quantization."""
    blocksize: int = 64
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.blocksize not in [64, 128, 256, 512]:
            raise ValueError(f"blocksize must be 64, 128, 256, or 512, got {self.blocksize}")


class NF4Quantizer:
    """
    NF4 Quantizer for 4-bit quantization with optional double quantization.

    This quantizer is compatible with bitsandbytes and optimized for Tesla T4.
    It uses block-wise quantization with configurable block sizes.

    Example:
        >>> quantizer = NF4Quantizer(blocksize=64, double_quant=True)
        >>> weight = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
        >>> qweight, state = quantizer.quantize(weight)
        >>> weight_restored = quantizer.dequantize(qweight, state)
    """

    def __init__(
        self,
        blocksize: int = 64,
        double_quant: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ):
        self.blocksize = blocksize
        self.double_quant = double_quant
        self.compute_dtype = compute_dtype

        # Create NF4 lookup tables
        self.nf4_table = torch.tensor(NF4_QUANT_TABLE, dtype=torch.float32)

    def quantize(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize a weight tensor to NF4 format.

        Args:
            weight: Input tensor to quantize (any shape)

        Returns:
            Tuple of:
                - quantized: uint8 tensor with 4-bit values packed
                - state: Dictionary containing quantization state
                  (absmax, code, offset for dequantization)
        """
        original_shape = weight.shape
        original_dtype = weight.dtype
        device = weight.device

        # Flatten weight for block-wise processing
        weight_flat = weight.flatten().float()
        n_elements = weight_flat.numel()

        # Pad to multiple of blocksize
        n_blocks = (n_elements + self.blocksize - 1) // self.blocksize
        padded_size = n_blocks * self.blocksize

        if padded_size > n_elements:
            weight_flat = torch.nn.functional.pad(
                weight_flat, (0, padded_size - n_elements), value=0
            )

        # Reshape into blocks
        weight_blocks = weight_flat.reshape(n_blocks, self.blocksize)

        # Compute absmax per block
        absmax = weight_blocks.abs().max(dim=1, keepdim=True)[0]
        absmax = torch.clamp(absmax, min=1e-8)  # Avoid division by zero

        # Normalize weights
        weight_normalized = weight_blocks / absmax

        # Quantize to NF4 using nearest neighbor
        nf4_table_device = self.nf4_table.to(device)
        quantized_blocks = torch.zeros_like(weight_blocks, dtype=torch.uint8)

        for i in range(n_blocks):
            for j in range(self.blocksize):
                val = weight_normalized[i, j]
                # Find nearest NF4 value
                distances = torch.abs(nf4_table_device - val)
                idx = torch.argmin(distances)
                quantized_blocks[i, j] = idx.item()

        # Pack two 4-bit values into one uint8
        quantized_flat = quantized_blocks.flatten()
        packed = torch.zeros(
            (quantized_flat.numel() + 1) // 2,
            dtype=torch.uint8,
            device=device
        )

        packed = (quantized_flat[0::2] << 4) | quantized_flat[1::2]

        # Create quantization state
        state = {
            'absmax': absmax.squeeze(-1),  # [n_blocks]
            'code': self.nf4_table.clone(),
            'blocksize': self.blocksize,
            'shape': original_shape,
            'dtype': original_dtype,
            'n_elements': n_elements,
        }

        # Double quantization for absmax if enabled
        if self.double_quant:
            state['absmax'], state['state2'] = self._double_quantize_absmax(
                state['absmax']
            )

        return packed, state

    def _double_quantize_absmax(
        self,
        absmax: torch.Tensor,
        blocksize2: int = 256,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply secondary quantization to absmax values."""
        n_blocks = absmax.numel()
        n_blocks2 = (n_blocks + blocksize2 - 1) // blocksize2
        padded_size2 = n_blocks2 * blocksize2

        if padded_size2 > n_blocks:
            absmax = torch.nn.functional.pad(
                absmax, (0, padded_size2 - n_blocks), value=0
            )

        absmax_blocks = absmax.reshape(n_blocks2, blocksize2)
        absmax2 = absmax_blocks.abs().max(dim=1, keepdim=True)[0]
        absmax2 = torch.clamp(absmax2, min=1e-8)

        # Quantize to uint8
        absmax_normalized = absmax_blocks / absmax2
        absmax_quantized = (absmax_normalized * 255.0).round().to(torch.uint8)

        state2 = {
            'absmax': absmax2.squeeze(-1),
            'code': torch.linspace(0, 1, 256, dtype=torch.float32),
            'blocksize': blocksize2,
        }

        return absmax_quantized.flatten()[:n_blocks], state2

    def dequantize(
        self,
        quantized: torch.Tensor,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Dequantize NF4 tensor back to original format.

        Args:
            quantized: Packed uint8 tensor
            state: Quantization state from quantize()

        Returns:
            Dequantized tensor in original shape and dtype
        """
        device = quantized.device

        # Unpack 4-bit values
        unpacked_size = quantized.numel() * 2
        unpacked = torch.zeros(unpacked_size, dtype=torch.uint8, device=device)
        unpacked[0::2] = (quantized >> 4) & 0x0F
        unpacked[1::2] = quantized & 0x0F

        # Get original dimensions
        n_elements = state['n_elements']
        blocksize = state['blocksize']
        n_blocks = (n_elements + blocksize - 1) // blocksize

        # Reconstruct absmax if double quantized
        absmax = state['absmax']
        if 'state2' in state:
            absmax = self._dequantize_absmax(absmax, state['state2'])

        absmax = absmax.to(device).float()

        # Get NF4 lookup table
        nf4_table = state['code'].to(device).float()

        # Dequantize blocks
        unpacked = unpacked[:n_blocks * blocksize]
        weight_blocks = unpacked.reshape(n_blocks, blocksize).float()

        # Map indices to NF4 values
        weight_normalized = nf4_table[weight_blocks.long()]

        # Denormalize
        weight_denorm = weight_normalized * absmax.unsqueeze(1)

        # Reshape and trim padding
        weight_flat = weight_denorm.flatten()[:n_elements]
        weight = weight_flat.reshape(state['shape'])

        # Convert to original dtype
        return weight.to(state['dtype'])

    def _dequantize_absmax(
        self,
        absmax_quantized: torch.Tensor,
        state2: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize the double-quantized absmax."""
        absmax2 = state2['absmax'].to(absmax_quantized.device).float()
        blocksize2 = state2['blocksize']
        n_blocks = absmax_quantized.numel()
        n_blocks2 = (n_blocks + blocksize2 - 1) // blocksize2

        # Dequantize uint8 to float
        absmax_norm = absmax_quantized.float() / 255.0

        # Pad and reshape
        padded_size = n_blocks2 * blocksize2
        if padded_size > n_blocks:
            absmax_norm = torch.nn.functional.pad(
                absmax_norm, (0, padded_size - n_blocks), value=0
            )

        absmax_blocks = absmax_norm.reshape(n_blocks2, blocksize2)

        # Denormalize
        absmax = absmax_blocks * absmax2.unsqueeze(1)

        return absmax.flatten()[:n_blocks]


def quantize_nf4(
    weight: torch.Tensor,
    blocksize: int = 64,
    double_quant: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quantize a tensor to NF4 format (convenience function).

    Args:
        weight: Tensor to quantize
        blocksize: Block size for quantization (64, 128, 256, or 512)
        double_quant: Enable double quantization of absmax values

    Returns:
        Tuple of (quantized_tensor, quantization_state)

    Example:
        >>> weight = torch.randn(4096, 11008, device='cuda', dtype=torch.float16)
        >>> qweight, state = quantize_nf4(weight, blocksize=64, double_quant=True)
        >>> print(f"Compression: {weight.nbytes / qweight.nbytes:.2f}x")
    """
    quantizer = NF4Quantizer(blocksize=blocksize, double_quant=double_quant)
    return quantizer.quantize(weight)


def dequantize_nf4(
    quantized: torch.Tensor,
    state: Dict[str, Any],
) -> torch.Tensor:
    """
    Dequantize NF4 tensor (convenience function).

    Args:
        quantized: Quantized tensor from quantize_nf4()
        state: Quantization state

    Returns:
        Dequantized tensor
    """
    quantizer = NF4Quantizer()
    return quantizer.dequantize(quantized, state)
