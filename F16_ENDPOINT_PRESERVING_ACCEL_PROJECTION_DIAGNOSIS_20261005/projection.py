"""Fixed preregistered projection. No ground truth or labels are accepted."""
import torch

DT = .4

def project(base, increments):
    assert base.shape == increments.shape and base.ndim == 4
    assert base.shape[1:] == (20, 12, 2)
    assert base.dtype == increments.dtype
    assert torch.isfinite(base).all() and torch.isfinite(increments).all()
    velocity = torch.diff(base, dim=2, prepend=torch.zeros_like(base[:, :, :1])) / DT
    native_acc = torch.diff(velocity, dim=2) / DT
    native_mean = torch.linalg.vector_norm(native_acc, dim=-1).mean((1, 2))
    delta_acc = torch.diff(increments, dim=2) / DT**2
    delta_mean = torch.linalg.vector_norm(delta_acc, dim=-1).mean((1, 2))
    # The D=0 branch is exactly specified; no new floor or epsilon.
    alpha = torch.ones_like(delta_mean)
    nonzero = delta_mean != 0
    alpha[nonzero] = (.01 * native_mean[nonzero] / delta_mean[nonzero]).clamp(0, 1)
    mean = increments.mean(2, keepdim=True)
    projected_u = mean + alpha[:, None, None, None] * (increments - mean)
    path = base + projected_u.cumsum(2)
    return path, projected_u, dict(alpha=alpha, native_mean=native_mean, delta_mean=delta_mean)
