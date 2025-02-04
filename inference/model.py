import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

class MLP(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_experts = args.n_routed_experts
        self.weight = nn.Parameter(torch.empty(self.n_experts, self.dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        scores = F.softmax(F.linear(x, self.weight), dim=-1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = torch.gather(scores, 1, indices)
        return weights, indices

class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(self.n_experts)])
    
    def forward(self, x):
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        for i in range(self.n_experts):
            mask = (indices == i).float().unsqueeze(-1)
            if mask.any():
                y += self.experts[i](x * mask) * weights.unsqueeze(-1)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = nn.MultiheadAttention(args.dim, args.num_heads, batch_first=True)
        self.ffn = MoE(args) if args.use_moe else MLP(args.dim, args.inter_dim)
        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = checkpoint(layer, x)
        return self.head(self.norm(x))

if __name__ == "__main__":
    class ModelArgs:
        vocab_size = 32000
        dim = 1024
        inter_dim = 4096
        n_layers = 12
        num_heads = 8
        use_moe = True
        n_routed_experts = 4
        n_activated_experts = 2
        moe_inter_dim = 4096
    
    args = ModelArgs()
    model = Transformer(args).cuda()
    x = torch.randint(0, args.vocab_size, (2, 128), device='cuda')
    print(model(x).size())
