_W8A8_EXPORTS = {
    "W8A8OF16LinearStaticScale",
    "W8A8OF16LinearDynamicInputScale",
    "FakeW8A8Linear",
    "fake_quant",
}

_SMOOTH_EXPORTS = {
    "get_act_scales",
    "get_smooth_scale",
    "smooth_ln_fcs",
    "smooth_fc_fc",
    "smooth_lm",
}

__all__ = sorted(_W8A8_EXPORTS | _SMOOTH_EXPORTS)


def __getattr__(name):
    if name in _W8A8_EXPORTS:
        from . import w8a8_linear

        value = getattr(w8a8_linear, name)
        globals()[name] = value
        return value

    if name in _SMOOTH_EXPORTS:
        from . import smooth

        value = getattr(smooth, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
