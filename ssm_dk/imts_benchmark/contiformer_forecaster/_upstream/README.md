# Vendored ContiFormer (subset of physiopro)

Source: https://github.com/microsoft/physiopro
Pinned commit at vendor time: latest of `main` as of 2026-04-21
License: MIT (Microsoft Corp.) — see [LICENSE](LICENSE)
Paper: Chen et al., "ContiFormer: Continuous-Time Transformer for Irregular
Time Series Modeling", NeurIPS 2023.
Reference standalone example: github.com/microsoft/SeqML/tree/main/ContiFormer

We vendor only what's needed for the model (no training framework):
```
physiopro/
├── network/contiformer.py     # MultiHeadAttention, EncoderLayer, ContiFormer
└── module/
    ├── linear.py              # ODELinear, InterpLinear
    ├── ode.py                 # build_fc_odefunc, TimeVariableODE
    ├── interpolate.py         # linear / hermite cubic interpolation coeffs
    └── positional_encoding.py # PositionalEncoding
```

**Patch applied**: removed `from .tsrnn import NETWORKS` and the
`@NETWORKS.register_module("contiformer")` decorator. The framework
registry depends on `microsoft/utilsd`, which we don't need — we
instantiate `ContiFormer` directly from our adapter.

**Pip deps added** (already installed in the project env): `torchdiffeq`,
`torchcde`. Both pulled in `torchsde`, `trampoline` as transitive.
