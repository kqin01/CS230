import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime

########################################
# Dataset
########################################
class PorousVelocityDataset(Dataset):
    def __init__(self, root_dir, pe_list=[0,10,20,50,100,200,500,1000,2000], u_scale=1000.0):
        self.image_dir = os.path.join(root_dir, "ims")
        self.u_dir = os.path.join(root_dir, "U")
        self.pe_list = pe_list
        self.u_scale = u_scale
        self.samples = [(img_id, pe) for img_id in range(500) for pe in self.pe_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, pe = self.samples[idx]
        batch_start = (img_id // 100) * 100
        batch_end = batch_start + 99
        batch_path = os.path.join(self.image_dir, f"{batch_start}-{batch_end}.pt")
        img_batch = torch.load(batch_path)
        image = img_batch[img_id % 100].float()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)

        u_path = os.path.join(self.u_dir, f"{img_id}_U_Pe{pe}.txt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (not os.path.exists(u_path)) or (os.path.getsize(u_path) == 0):
                return None
            u_vec_full = np.loadtxt(u_path)
        if u_vec_full.size < 2:
            return None

        u_vec = torch.tensor(u_vec_full[:2], dtype=torch.float32) / self.u_scale
        pe_tensor = torch.tensor([pe / 2000.0], dtype=torch.float32)
        return image, pe_tensor, u_vec


########################################
# CNN Model
########################################
class PorousCNN_U(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(65, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, image, pe):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, pe], dim=1)
        return self.fc(x)


########################################
# Collate function
########################################
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data._utils.collate.default_collate(batch)


########################################
# Training
########################################
def train_with_plot(
    root_dir="/home/kqin/OpenFOAM/kqin-7/quickCNN",
    epochs=20,
    batch_size=16,
    lr=1e-4
):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = PorousVelocityDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none)

    # Initialize model, optimizer, loss
    model = PorousCNN_U().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Logging
    loss_history = []
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(root_dir, f"train_log_{start_time}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Loss\n")

    print("----- Training Start -----")

    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            image, pe, target_u = batch
            image, pe, target_u = image.to(device), pe.to(device), target_u.to(device)

            pred = model(image, pe)
            loss = loss_fn(pred, target_u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * image.size(0)
            count += image.size(0)

        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)

        # Logging and display
        print(f"Epoch {epoch+1:02d}/{epochs}  Loss: {avg_loss:.6f}")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f}\n")

    print("✅ Training complete.")
    model_path = os.path.join(root_dir, "cnn_u_train_plot.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved at: {model_path}")

    # Plot training curve
    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(root_dir, f"loss_curve_{start_time}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"📊 Loss curve saved at: {plot_path}")


if __name__ == "__main__":
    train_with_plot()