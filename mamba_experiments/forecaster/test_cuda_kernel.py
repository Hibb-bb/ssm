"""
Quick validation that the mamba_ssm CUDA kernel works correctly.
Tests:
  1. selective_scan_cuda loads and runs (forward + backward)
  2. CUDA kernel output matches the pure-PyTorch reference
  3. Full Mamba block forward pass works
  4. Benchmark: CUDA kernel vs PyTorch-loop scan speed
"""
import time
import torch
import torch.nn.functional as F

def test_selective_scan_cuda():
    """Test that the CUDA selective scan kernel loads, runs, and matches reference."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    torch.manual_seed(42)
    device = "cuda"
    batch, dim, seqlen, dstate = 4, 64, 512, 16

    u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)
    delta = F.softplus(torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True))
    A = -torch.rand(dim, dstate, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(batch, dstate, seqlen, device=device, dtype=torch.float32, requires_grad=True)
    C = torch.randn(batch, dstate, seqlen, device=device, dtype=torch.float32, requires_grad=True)
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32, requires_grad=True)

    # Reference (pure PyTorch)
    out_ref = selective_scan_ref(
        u.detach().clone().requires_grad_(),
        delta.detach().clone().requires_grad_(),
        A.detach().clone().requires_grad_(),
        B.detach().clone().requires_grad_(),
        C.detach().clone().requires_grad_(),
        D=D.detach().clone().requires_grad_(),
        z=z.detach().clone().requires_grad_(),
        delta_softplus=False,
    )

    # CUDA kernel
    out_cuda = selective_scan_fn(
        u, delta, A, B, C, D=D, z=z, delta_softplus=False,
    )

    max_diff = (out_ref - out_cuda).abs().max().item()
    print(f"[1] selective_scan: CUDA vs ref max diff = {max_diff:.2e}  ", end="")
    assert max_diff < 1e-3, f"CUDA kernel output differs from reference: {max_diff}"
    print("PASS")

    # Test backward
    loss = out_cuda.sum()
    loss.backward()
    print(f"[2] backward pass:  ", end="")
    assert u.grad is not None and u.grad.abs().sum() > 0
    print("PASS")


def test_mamba_block():
    """Test a full Mamba block forward + backward."""
    from mamba_ssm import Mamba

    device = "cuda"
    batch, seqlen, d_model = 4, 512, 256

    model = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device)
    x = torch.randn(batch, seqlen, d_model, device=device, requires_grad=True)

    y = model(x)
    print(f"[3] Mamba block:    input {x.shape} -> output {y.shape}  ", end="")
    assert y.shape == x.shape
    print("PASS")

    loss = y.sum()
    loss.backward()
    print(f"[4] Mamba backward: ", end="")
    assert x.grad is not None
    print("PASS")


def benchmark_scan():
    """Benchmark CUDA scan vs pure-PyTorch sequential loop."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    device = "cuda"
    batch, dim, dstate = 128, 768, 16

    for seqlen in [160, 512, 2048]:
        u = torch.randn(batch, dim, seqlen, device=device)
        delta = F.softplus(torch.randn(batch, dim, seqlen, device=device))
        A = -torch.rand(dim, dstate, device=device)
        B = torch.randn(batch, dstate, seqlen, device=device)
        C = torch.randn(batch, dstate, seqlen, device=device)
        D = torch.randn(dim, device=device)

        # Warmup
        for _ in range(3):
            selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=False)
        torch.cuda.synchronize()

        # CUDA kernel timing
        n_iter = 20
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=False)
        torch.cuda.synchronize()
        cuda_ms = (time.time() - t0) / n_iter * 1000

        # PyTorch loop timing
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-2))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1).unsqueeze(-1)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            x = torch.zeros(batch, dim, dstate, device=device)
            ys = []
            for i in range(seqlen):
                x = deltaA[:, :, i] * x + deltaB[:, :, i] * u[:, :, i].unsqueeze(-1)
                y_i = (x * C[:, :, i].unsqueeze(1)).sum(-1)
                ys.append(y_i)
            torch.stack(ys, dim=-1)
        torch.cuda.synchronize()
        loop_ms = (time.time() - t0) / n_iter * 1000

        speedup = loop_ms / cuda_ms
        print(f"[5] L={seqlen:>5d}: CUDA={cuda_ms:7.2f}ms  PyLoop={loop_ms:7.2f}ms  speedup={speedup:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Mamba CUDA Kernel Validation & Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    test_selective_scan_cuda()
    test_mamba_block()
    print()
    print("--- Benchmark: CUDA kernel vs Python loop ---")
    print(f"    (batch=128, d_inner=768, d_state=16, {20} iterations)")
    benchmark_scan()
    print()
    print("All tests passed!")
