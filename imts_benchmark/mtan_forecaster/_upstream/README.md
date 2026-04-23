# Vendored mTAN

Source: https://github.com/reml-lab/mTAN
Pinned commit at vendor time: latest of `main` as of 2026-04-21
License: MIT (copyright 2020 Satya Narayan Shukla) — see [LICENSE](LICENSE)
Paper: Shukla & Marlin, "Multi-Time Attention Networks for Irregularly
Sampled Time Series", ICLR 2021. https://openreview.net/forum?id=4c0J6lwQ4_

Only `models.py` is vendored. Of the classes inside, our adapter uses
`multiTimeAttention`, `enc_mtan_rnn`, and `dec_mtan_rnn`. The classification
heads / interpolation variants are unused but kept verbatim for upstream
fidelity.

The upstream `models.py` hardcodes `self.device = device` at init and uses it
to move time-embedding tensors. Our adapter monkey-patches `self.device` at
forward time so Lightning's device-placement remains the source of truth.
