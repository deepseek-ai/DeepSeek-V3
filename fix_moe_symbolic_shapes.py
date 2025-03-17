import torch
import torch._dynamo

# Solution 1: Suppress errors (quick fix but not recommended for production)
torch._dynamo.config.suppress_errors = True

# Solution 2: Example of a more robust way to handle MoE with dynamic shapes
class RobustMoE(torch.nn.Module):
    def __init__(self, num_experts, d_model):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])
        self.router = torch.nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # Get routing weights
        route_weights = torch.softmax(self.router(x), dim=-1)
        
        # Instead of using if conditions on counts, use masked operations
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            # Apply expert computation to all inputs
            expert_out = self.experts[i](x)
            # Weight the outputs by routing weights
            outputs += route_weights[..., i:i+1] * expert_out
        
        return outputs

"""
Usage example:
model = RobustMoE(num_experts=4, d_model=256)
x = torch.randn(32, 256)  # batch_size=32, d_model=256
output = model(x)

This implementation avoids the GuardOnDataDependentSymNode error by:
1. Not using data-dependent control flow (if statements based on counts)
2. Using masked operations instead
3. If needed, you can still enable error suppression with:
   torch._dynamo.config.suppress_errors = True
"""