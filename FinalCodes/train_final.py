import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings


########################################
# USER CONFIG
########################################
DATA_ROOT   = "/home/kqin/OpenFOAM/kqin-7/quickDeepCNN5000"
IMS_DIR     = "/home/kqin/OpenFOAM/kqin-7/ims"
U_DIR       = "/home/kqin/OpenFOAM/kqin-7/run/data2000Da1/U"

NUM_IMAGES  = 5000
PE_LIST     = [0, 10] # Pe number choice, Da = 1 here
U_SCALE     = 10.0

EPOCHS      = 200
BATCH       = 64
LR          = 1e-4


########################################
# Dataset
########################################
class PorousVelocityDataset(Dataset):
    def __init__(self, ims_dir, u_dir, split, pe_list, u_scale, num_images):

        self.ims_dir = ims_dir
        self.u_dir   = u_dir
        self.pe_list = pe_list
        self.u_scale = u_scale

        all_ids = np.arange(num_images)

        n_train = int(num_images * 0.8)
        n_dev   = int(num_images * 0.1)

        if split == "train":
            self.img_ids = all_ids[:n_train]
        elif split == "dev":
            self.img_ids = all_ids[n_train:n_train+n_dev]
        elif split == "test":
            self.img_ids = all_ids[n_train+n_dev:]
        else:
            raise ValueError("split must be train/dev/test")

        # Only include samples whose U file exists and is not empty
        self.samples = []
        for img_id in self.img_ids:
            for pe in self.pe_list:
                u_file = os.path.join(self.u_dir, f"{img_id}_U_Pe{pe}.txt")
                if os.path.exists(u_file) and os.path.getsize(u_file) > 5:
                    self.samples.append((img_id, pe))

        print(f"[{split}] Loaded {len(self.samples)} samples (Pe={self.pe_list})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_id, pe = self.samples[idx]

        batch_start = (img_id // 100) * 100
        batch_file  = os.path.join(self.ims_dir, f"{batch_start}-{batch_start+99}.pt")

        img_batch = torch.load(batch_file)
        image = img_batch[img_id % 100].float()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        u_file = os.path.join(self.u_dir, f"{img_id}_U_Pe{pe}.txt")
        u_full = np.loadtxt(u_file)
        u_vec = torch.tensor(u_full[:2], dtype=torch.float32) / self.u_scale

        pe_norm = torch.tensor([pe / max(self.pe_list)], dtype=torch.float32)

        return image, pe_norm, u_vec



########################################
# CNN Model  (deep + small dropout)
########################################
class PorousCNN_U(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, image, pe):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, pe], dim=1)
        return self.fc(x)



########################################
# Color helper
########################################
def get_pe_colors(pe_values):
    """Return a dict {Pe: color} with fixed colors for 0 and 10 and
    automatic tab20 colors for others."""
    pe_values = sorted(list(set(pe_values)))
    base = {
        0:  "#80e5e5",  # cyan
        10: "#7f7f7f",  # gray
    }
    cmap = plt.get_cmap("tab20")
    colors = {}
    j = 0
    for pe in pe_values:
        if pe in base:
            colors[pe] = base[pe]
        else:
            colors[pe] = cmap(j % cmap.N)
            j += 1
    return colors



########################################
# Visualization
########################################
def plot_pred_vs_true_split(model, loaders, names, device, u_scale, save_prefix):

    model.eval()

    for loader, name in zip(loaders, names):

        preds, trues, pes_true = [], [], []

        with torch.no_grad():
            for image, pe_norm, target in loader:
                pred = model(image.to(device), pe_norm.to(device))

                preds.append(pred.cpu().numpy() * u_scale)
                trues.append(target.numpy() * u_scale)

                # Restore original Pe from normalized value
                pes_true.append(pe_norm.numpy().reshape(-1) * max(PE_LIST))

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        pes_true = np.concatenate(pes_true)

        unique_pes = sorted(np.unique(pes_true))
        pe_colors = get_pe_colors(unique_pes)

        # ---------------- Ux ----------------
        plt.figure(figsize=(6, 6))
        for pe in unique_pes:
            mask = (pes_true == pe)
            plt.scatter(trues[mask, 0], preds[mask, 0],
                        s=14, alpha=0.6, color=pe_colors[pe],
                        label=f"Pe={int(pe)}")

        mn = min(trues[:,0].min(), preds[:,0].min())
        mx = max(trues[:,0].max(), preds[:,0].max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)

        plt.xlabel("True Ux")
        plt.ylabel("Pred Ux")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{name}_Ux.png", dpi=150)
        plt.close()

        # ---------------- Uy ----------------
        plt.figure(figsize=(6, 6))
        for pe in unique_pes:
            mask = (pes_true == pe)
            plt.scatter(trues[mask, 1], preds[mask, 1],
                        s=14, alpha=0.6, color=pe_colors[pe],
                        label=f"Pe={int(pe)}")

        mn = min(trues[:,1].min(), preds[:,1].min())
        mx = max(trues[:,1].max(), preds[:,1].max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)

        plt.xlabel("True Uy")
        plt.ylabel("Pred Uy")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{name}_Uy.png", dpi=150)
        plt.close()



########################################
# Training
########################################
def train_full():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    train_set = PorousVelocityDataset(IMS_DIR, U_DIR, "train", PE_LIST, U_SCALE, NUM_IMAGES)
    dev_set   = PorousVelocityDataset(IMS_DIR, U_DIR, "dev",   PE_LIST, U_SCALE, NUM_IMAGES)
    test_set  = PorousVelocityDataset(IMS_DIR, U_DIR, "test",  PE_LIST, U_SCALE, NUM_IMAGES)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True)
    dev_loader   = DataLoader(dev_set,   batch_size=BATCH)
    test_loader  = DataLoader(test_set,  batch_size=BATCH)

    model = PorousCNN_U().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("---- Training ----")
    train_losses, dev_losses = [], []

    for ep in range(EPOCHS):

        # Training loop
        model.train()
        total, cnt = 0, 0

        for img, pe, u in train_loader:
            img, pe, u = img.to(device), pe.to(device), u.to(device)
            pred = model(img, pe)
            loss = loss_fn(pred, u)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(img)
            cnt   += len(img)
        train_losses.append(total/cnt)

        # Dev loop
        model.eval()
        total, cnt = 0, 0
        with torch.no_grad():
            for img, pe, u in dev_loader:
                pred = model(img.to(device), pe.to(device))
                loss = loss_fn(pred, u.to(device))
                total += loss.item() * len(img)
                cnt   += len(img)
        dev_losses.append(total/cnt)

        print(f"Epoch {ep+1}/{EPOCHS} | Train={train_losses[-1]:.4f} | Dev={dev_losses[-1]:.4f}")

    # ---------------- Test phase ----------------
    total, cnt = 0, 0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for img, pe, u in test_loader:
            pred = model(img.to(device), pe.to(device))
            all_preds.append(pred.cpu().numpy() * U_SCALE)
            all_trues.append(u.numpy() * U_SCALE)
            loss = loss_fn(pred, u.to(device))
            total += loss.item() * len(img)
            cnt   += len(img)

    test_loss = total / cnt
    print(f"Final Test Loss = {test_loss:.6f}")


    # # ---------------- Save model ----------------
    # model_path = os.path.join(DATA_ROOT, "model_final.pth")
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")
    # # --------------------------------------------


    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    # Save predictions + truth
    np.savetxt(os.path.join(DATA_ROOT, "predictions.txt"), all_preds)
    np.savetxt(os.path.join(DATA_ROOT, "truth.txt"), all_trues)

    # RMS
    rms = np.sqrt(np.mean((all_preds - all_trues)**2, axis=0))

    with open(os.path.join(DATA_ROOT, "metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"RMS Ux: {rms[0]}\n")
        f.write(f"RMS Uy: {rms[1]}\n")

    # Save loss curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(dev_losses, label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_ROOT, "loss_curve.png"), dpi=150)
    plt.close()

    # Scatter plots for each split
    plot_pred_vs_true_split(
        model,
        [train_loader, dev_loader, test_loader],
        ["train", "dev", "test"],
        device,
        U_SCALE,
        os.path.join(DATA_ROOT, "UxUy")
    )


if __name__ == "__main__":
    train_full()