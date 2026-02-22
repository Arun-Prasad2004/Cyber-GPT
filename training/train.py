import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os

from model.cybergpt import CyberGPT
from training.dataset import CyberDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Fine-tuning Config
# ========================
batch_size = 4
block_size = 512
lr = 2e-5          # MUCH lower LR for smoothing
grad_accum_steps = 8
max_additional_steps = 20000   # only fine-tune 20k more
save_interval = 2000

# ========================
# Dataset
# ========================
dataset = CyberDataset(
    "data/processed/shuffled_corpus.txt",
    "tokenizer/cyber_tokenizer.json",
    block_size=block_size
)

loader = DataLoader(dataset, batch_size=batch_size)

# ========================
# Model
# ========================
model = CyberGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler("cuda")

start_step = 0

# ========================
# Load Checkpoint (Required)
# ========================
if not os.path.exists("cybergpt_checkpoint.pt"):
    raise RuntimeError("Checkpoint not found. Fine-tuning requires existing trained model.")

print("Loading checkpoint for fine-tuning...")

checkpoint = torch.load("cybergpt_checkpoint.pt", map_location=device)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint.get("step", 0)
else:
    model.load_state_dict(checkpoint)

print(f"Starting fine-tuning from step {start_step}")

model.train()

# ========================
# Safe Save Function
# ========================
def safe_save(step):
    print(f"\nSaving checkpoint at step {step}...")

    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    torch.save({
        "model_state": cpu_state_dict,
        "optimizer_state": optimizer.state_dict(),
        "step": step
    }, "cybergpt_checkpoint.pt")

    print("Checkpoint saved successfully.")

# ========================
# Fine-Tuning Loop
# ========================
global_step = start_step
target_step = start_step + max_additional_steps

try:
    pbar = tqdm(loader)

    for x, y in pbar:

        if global_step >= target_step:
            break

        x, y = x.to(device), y.to(device)

        with autocast("cuda"):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        scaler.scale(loss / grad_accum_steps).backward()

        if (global_step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pbar.set_description(f"FineTune Step {global_step} | Loss: {loss.item():.4f}")

        if global_step % 1000 == 0:
            torch.cuda.empty_cache()

        if global_step % save_interval == 0 and global_step != 0:
            safe_save(global_step)

        global_step += 1

except KeyboardInterrupt:
    print("\nFine-tuning interrupted.")

finally:
    safe_save(global_step)
    print(f"Fine-tuning stopped at step {global_step}")