import torch
from kernel import decode  # Assuming kernel.py is in the same folder
from model import DummyTransformer  # The dummy transformer we just created

# Instantiate the dummy model
model = DummyTransformer()

# Define a sample input (a small sequence of token IDs, e.g., from GPT tokenizer)
input_ids = torch.randint(0, 50257, (1, 10))  # Batch size of 1, sequence length of 10
position = 5  # We are generating the next token at position 5

# Call the decode function
logits_or_probs = decode(
    input_ids=input_ids,
    position=position,
    model=model,
    apply_softmax=True,  # Toggle softmax to get probabilities instead of raw logits
    top_k=10,  # Set top-k filtering
    top_p=0.9,  # Set top-p filtering (nucleus sampling)
    device='cpu'  # Can switch to 'cuda' if you have a GPU
)

# Print the output
print("Output probabilities (softmax applied):")
print(logits_or_probs)
