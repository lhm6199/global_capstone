import torch
import torch.nn.functional as F

from awq.quantize.qmodule import WQLinear


def unpack_awq_int4_weight(qweight, out_features, in_features):
    """Unpack TinyChat/AWQ interleaved INT4 weights to [out_features, in_features]."""
    if in_features % 64 != 0:
        raise ValueError("AWQ packed INT4 weights expect in_features to be divisible by 64.")
    if out_features % 4 != 0:
        raise ValueError("AWQ packed INT4 weights expect out_features to be divisible by 4.")

    packed = qweight.to(torch.int32) & 0xFFFF
    unpacked = torch.stack(
        [(packed >> shift) & 0xF for shift in (0, 4, 8, 12)],
        dim=-1,
    )

    # Inverse of pack_intweight(..., interleave=4, kstride=64).
    unpacked = unpacked.reshape(out_features // 4, in_features // 64, 64, 4)
    unpacked = unpacked.reshape(out_features // 4, in_features // 64, 4, 64)
    unpacked = unpacked.transpose(1, 2).reshape(out_features, in_features)

    # Inverse of the two intra-row reorder steps used by pack_intweight.
    unpacked = unpacked.reshape(out_features, in_features // 32, 4, 2, 4)
    unpacked = unpacked.transpose(3, 4).reshape(out_features, in_features // 32, 4, 8)
    unpacked = unpacked.reshape(out_features, in_features // 32, 4, 4, 2)
    unpacked = unpacked.transpose(2, 3).reshape(out_features, in_features)
    return unpacked


def dequantize_awq_weight(module):
    qweight = unpack_awq_int4_weight(
        module.qweight,
        out_features=module.out_features,
        in_features=module.in_features,
    ).to(module.scales.dtype)

    group_ids = torch.arange(module.in_features, device=module.scales.device)
    group_ids = torch.div(group_ids, module.group_size, rounding_mode="floor")
    scales = module.scales.index_select(0, group_ids).transpose(0, 1).contiguous()
    scaled_zeros = module.scaled_zeros.index_select(0, group_ids).transpose(0, 1)
    return qweight * scales + scaled_zeros


@torch.no_grad()
def torch_wqlinear_forward(self, x):
    weight = getattr(self, "_torch_dequantized_weight", None)
    if weight is None:
        weight = dequantize_awq_weight(self)
    return F.linear(x, weight.to(dtype=x.dtype), self.bias)


@torch.no_grad()
def cache_dequantized_weights(model):
    for module in model.modules():
        if isinstance(module, WQLinear):
            module.register_buffer(
                "_torch_dequantized_weight",
                dequantize_awq_weight(module),
                persistent=False,
            )


def enable_wqlinear_torch_forward(model=None, cache_weights=False):
    WQLinear.forward = torch_wqlinear_forward
    if model is not None and cache_weights:
        cache_dequantized_weights(model)
    return model
