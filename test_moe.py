import torch
from fix_moe_symbolic_shapes import RobustMoE

def test_moe():
    # Test with both default behavior and compiled version
    model = RobustMoE(num_experts=4, d_model=256)
    x = torch.randn(32, 256)  # batch_size=32, d_model=256
    
    # Test 1: Regular forward pass
    print("Testing regular forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test 2: Compiled version
    print("\nTesting compiled version...")
    compiled_model = torch.compile(model)
    compiled_output = compiled_model(x)
    print(f"Compiled output shape: {compiled_output.shape}")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_moe()