import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from core.modules.so3.dist import sample_matrix_fisher_torch
from core.dataset import so3_utils

import numpy as np
import math

LOG2PI = np.log(2 * np.pi)


def uniform_SO3_torch(n, device='cpu'):
    u1, u2, u3 = torch.rand(3, n, device=device)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    R = torch.zeros((n,3,3), device=device)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

def exp_so3(omega):
    theta = torch.norm(omega, dim=1, keepdim=True) + 1e-12
    k = omega / theta
    K = torch.zeros((omega.shape[0], 3, 3), device=omega.device)
    K[:,0,1],K[:,0,2],K[:,1,0],K[:,1,2],K[:,2,0],K[:,2,1] = -k[:,2],k[:,1],k[:,2],-k[:,0],-k[:,1],k[:,0]
    I = torch.eye(3, device=omega.device).unsqueeze(0)
    sin_term = torch.sin(theta)[:,None] * K
    cos_term = (1-torch.cos(theta))[:,None] * torch.bmm(K,K)
    return I + sin_term + cos_term


class BFNBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BFNBase, self).__init__(*args, **kwargs)

    def trans_bayesian_update(self, t, sigma1, x):
        gamma = 1 - torch.pow(sigma1, 2 * t[:,None])
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        mu = gamma * x
        return mu, gamma

    def sample_matrix_fisher_mixed(self, lambda_val=25, n_samples=10000, device='cpu'):
        lambda_val = float(lambda_val)
        if lambda_val <= 26:
            samples = []
            batch = max(2000, n_samples * 5)
            max_density = torch.exp(torch.tensor(3.0 * lambda_val, device=device))
            total = 0
            while total < n_samples:
                R = uniform_SO3_torch(batch, device=device)
                tr = torch.einsum('bii->b', R)
                density = torch.exp(lambda_val * tr)
                u = torch.rand(batch, device=device)
                accept = R[u < density / max_density]
                if accept.numel() > 0:
                    samples.append(accept)
                    total += accept.shape[0]
            R_samples = torch.cat(samples, dim=0)[:n_samples]
        else:
            sigma = 1.0 / math.sqrt(2*lambda_val)
            omega = torch.randn(n_samples, 3, device=device) * sigma
            R_samples = exp_so3(omega)
        return R_samples

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        beta = beta1 * (t[:,None]**2)
        one_hot_x = x
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        y = mean
        theta = F.softmax(y, dim=-1)
        return theta

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, mask):
        weight = N * (1 - sigma1**(2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        loss = weight.view(-1) * (((x_pred - x) ** 2).sum(-1)*mask).sum(-1)/mask.sum(-1)
        return loss.mean()

    def get_lambdat(self, t, lambda1):
        return lambda1 / (math.exp(2) - 1) * (torch.exp(2 * t) - 1)

    def dtime4so3_loss(self, i, N, lambda1, x_pred, x, mask):
        weight = self.get_lambdat(i/N, lambda1)
        weight = weight * (1-1/(2*weight+1))
        R_rel = torch.matmul(x.transpose(-2, -1), x_pred)
        dist = 3-torch.einsum('njii->nj', R_rel)
        dist = weight * dist
        dist = (dist*mask).sum(-1)/mask.sum(-1)
        return N * dist.mean()

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, mask=None
    ):
        target_x = one_hot_x
        e_hat = p_0
        alpha = beta1 * (2 * i - 1) / N**2
        alpha = alpha.view(-1, 1)
        classes = torch.arange(K, device=target_x.device).long()[None,None,...]
        e_x = F.one_hot(classes.long(), K)
        receiver_components = dist.Independent(
            dist.Normal(
                alpha[...,None,None] * ((K * e_x) - 1),
                (K * alpha[...,None,None]) ** 0.5,
            ),
            1,
        )
        receiver_mix_distribution = dist.Categorical(probs=e_hat)
        receiver_dist = dist.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )
        sender_dist = dist.Independent( dist.Normal(
            alpha[...,None]* ((K * target_x) - 1), ((K * alpha)[...,None] ** 0.5)
        ),1)
        y = sender_dist.sample(torch.Size([n_samples]))
        loss = N * ((sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0)*mask).sum() / mask.sum()
        return loss

    def factored_discrete_update(self, t, beta1, x, K=4):
        """
        Factored discrete BFN posterior update for FSQ codes.
        x: (B, L, D, K) one-hot or soft targets, D=6 factored dims, K=4 levels
        Returns: (B, L, D, K) posterior probabilities per dimension
        """
        B, L, D, Kd = x.shape
        x_flat = x.reshape(B, L * D, Kd)
        theta_flat = self.discrete_var_bayesian_update(t, beta1, x_flat, Kd)
        return theta_flat.reshape(B, L, D, Kd)

    def factored_discrete_loss(self, i, N, beta1, one_hot_x, p_0, K=4, n_samples=200, mask=None):
        """
        Factored discrete BFN loss: sum of per-dimension K=4 discrete losses.
        one_hot_x: (B, L, D, K) ground truth one-hot per dimension
        p_0:       (B, L, D, K) predicted probabilities per dimension
        mask:      (B, L) valid residue mask
        Returns: scalar loss
        """
        B, L, D, Kd = one_hot_x.shape
        one_hot_flat = one_hot_x.reshape(B, L * D, Kd)
        p_0_flat = p_0.reshape(B, L * D, Kd)
        if mask is not None:
            mask_flat = mask.unsqueeze(-1).expand(B, L, D).reshape(B, L * D)
        else:
            mask_flat = None
        return self.dtime4discrete_loss_prob(
            i, N, beta1, one_hot_flat, p_0_flat, Kd, n_samples=n_samples, mask=mask_flat
        )

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
