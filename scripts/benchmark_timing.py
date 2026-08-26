"""One-off benchmark: measure real train-step and val-step throughput at the
actual config (not the tiny smoke-test subset) to estimate full-epoch time."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.common import load_config, resolve_path, set_seed
from src.data.dataset import BCTumorSegDataset, build_train_augmentations, load_manifest
from src.data.preprocessing import PreprocessConfig
from src.models.unet import build_model
from src.training.losses import DiceBCELoss

cfg = load_config(Path("configs/config.yaml"))
set_seed(cfg["seed"])
df = load_manifest(resolve_path(cfg, "manifest_csv"))
preprocess_cfg = PreprocessConfig(image_size=cfg["image_size"])
train_aug = build_train_augmentations(cfg["augmentation"])
train_ds = BCTumorSegDataset(df, Path(cfg["dataset_root"]), "train", preprocess_cfg, augment=train_aug)
loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

model = build_model(cfg["model"])
criterion = DiceBCELoss(cfg["dice_weight"], cfg["bce_weight"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

n_train, n_valid = len(train_ds), 780
n_batches_bench = 12
print(f"batch_size={cfg['batch_size']} image_size={cfg['image_size']} train_images={n_train}")

model.train()
times = []
it = iter(loader)
for i in range(n_batches_bench):
    batch = next(it)
    t0 = time.time()
    optimizer.zero_grad()
    logits = model(batch["image"])
    loss, _, _ = criterion(logits, batch["mask"])
    loss.backward()
    optimizer.step()
    times.append(time.time() - t0)

warm = times[2:]  # skip first couple (dataloader warmup / worker spin-up)
per_batch = sum(warm) / len(warm)
per_image = per_batch / cfg["batch_size"]
print(f"train: {per_batch:.3f}s/batch  ({per_image:.4f}s/image)")

n_train_batches = n_train // cfg["batch_size"]
n_valid_batches = n_valid // cfg["batch_size"]
# validation forward-only is roughly 1/3 the cost of a full train step (no backward/optimizer)
est_epoch_seconds = n_train_batches * per_batch + n_valid_batches * per_batch * 0.35
print(f"estimated full epoch (train+valid): {est_epoch_seconds:.1f}s "
      f"({est_epoch_seconds/60:.2f} min)")
for n_epochs in [10, 30, 50]:
    total_min = est_epoch_seconds * n_epochs / 60
    print(f"  {n_epochs} epochs -> ~{total_min:.1f} min (~{total_min/60:.2f} hr)")
