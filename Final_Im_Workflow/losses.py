import numpy as np
import matplotlib.pyplot as plt

save_every = 3
iterations = list(range(save_every, 37, save_every))
noise_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

for noise in noise_levels:

    prefix = "clean" if noise == 0 else f"noisy{noise}"
    mlem_prefix = '' if noise == 0 else f"noisy{noise}_"

    for it in iterations:

        # Load
        train_loss = np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/{prefix}_extra{it}_train_loss_final.npy")
        val_loss = np.load(f"/scratch/bggjem001/picoPET-sim/Final_Im_Workflow/model_weights/{prefix}_extra{it}_val_loss_final.npy")

        # Plot
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train loss")
        plt.plot(epochs, val_loss, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{prefix} — extra iteration {it}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_extra{it}_loss_final.png", dpi=150)
        plt.show()