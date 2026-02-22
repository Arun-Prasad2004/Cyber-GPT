import torch
from tokenizers import Tokenizer
from model.cybergpt import CyberGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Load tokenizer
# ========================
tokenizer = Tokenizer.from_file("tokenizer/cyber_tokenizer.json")

# ========================
# Load model
# ========================
model = CyberGPT().to(device)

checkpoint = torch.load("cybergpt_checkpoint.pt", map_location=device, weights_only=False)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# ========================
# Generation Function
# ========================
def generate(
    prompt,
    max_new_tokens=150,
    temperature=0.9,
    top_k=50,
    repetition_penalty=1.2
):
    tokens = tokenizer.encode(prompt).ids
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        with torch.no_grad():
            logits = model(tokens)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Repetition penalty
            for token_id in set(tokens[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty

            # Top-k sampling
            probs = torch.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)

            next_token = top_k_indices[
                0,
                torch.multinomial(top_k_probs[0], 1)
            ].unsqueeze(0)

        tokens = torch.cat([tokens, next_token], dim=1)

        if tokens.size(1) > 512:
            tokens = tokens[:, -512:]

    return tokenizer.decode(tokens[0].tolist())


# ========================
# Test Prompt
# ========================
prompt = "Explain SQL injection vulnerability."
output = generate(prompt)

print("\n=== Generated Output ===\n")
print(output)