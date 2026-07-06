from __future__ import annotations

import torch


def normalize_vector(v: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v: torch.Tensor, e: torch.Tensor, dim: int) -> torch.Tensor:
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)


def local_to_global(R: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if p.size(-1) != 3:
        raise ValueError("Expected local coordinates with trailing dimension 3.")

    p_size = p.size()
    batch_size, length = p_size[0], p_size[1]
    p_flat = p.view(batch_size, length, -1, 3).transpose(-1, -2)
    q = torch.matmul(R, p_flat) + t.unsqueeze(-1)
    return q.transpose(-1, -2).reshape(p_size)


def global_to_local(R: torch.Tensor, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.size(-1) != 3:
        raise ValueError("Expected global coordinates with trailing dimension 3.")

    q_size = q.size()
    batch_size, length = q_size[0], q_size[1]
    q_flat = q.reshape(batch_size, length, -1, 3).transpose(-1, -2)
    p = torch.matmul(R.transpose(-1, -2), q_flat - t.unsqueeze(-1))
    return p.transpose(-1, -2).reshape(q_size)
