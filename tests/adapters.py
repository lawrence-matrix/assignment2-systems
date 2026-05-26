from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        d = q.shape[-1]
        scale = d ** -0.5

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            nq = q.shape[-2]
            nk = k.shape[-2]
            mask = (
                torch.arange(nq, device=q.device)[:, None]
                >= torch.arange(nk, device=q.device)[None, :]
            )
            scores = torch.where(mask, scores, torch.full_like(scores, -1e6))

        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        ctx.save_for_backward(q, k, v, probs, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, probs, lse = ctx.saved_tensors

        dv = torch.matmul(probs.transpose(-2, -1), grad_out)

        dp = torch.matmul(grad_out, v.transpose(-2, -1))

        ds = probs * (dp - (dp * probs).sum(dim=-1, keepdim=True))

        scale = q.shape[-1] ** -0.5

        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale

        return dq, dk, dv, None


class FlashAttentionTriton(FlashAttentionPyTorch):
    pass


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


class SimpleDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        seen = set()
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue

            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)

            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


class SimpleFSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        if self.compute_dtype is None:
            return self.module(*args, **kwargs)

        args = [
            a.to(self.compute_dtype) if torch.is_floating_point(a) else a
            for a in args
        ]

        orig = {}
        for name, p in self.module.named_parameters():
            orig[name] = p.data
            if torch.is_floating_point(p.data):
                p.data = p.data.to(self.compute_dtype)

        out = self.module(*args, **kwargs)

        for name, p in self.module.named_parameters():
            p.data = orig[name]

        return out

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        seen = set()
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue

            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)

            if p.grad.dtype != torch.float32:
                p.grad = p.grad.float()

            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


class ShardedOptimizer:
    def __init__(self, params, optimizer_cls, **kwargs):
        self.params = list(params)
        self.optimizer = optimizer_cls(self.params, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    @torch.no_grad()
    def step(self, *args, **kwargs):
        out = self.optimizer.step(*args, **kwargs)

        for p in self.params:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= dist.get_world_size()

        return out

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state



def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return SimpleDDP(module)



def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ddp_model.finish_gradient_synchronization()



def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    return SimpleFSDP(module, compute_dtype=compute_dtype)



def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()



def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in fsdp_model.module.state_dict().items()
    }



def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
```


