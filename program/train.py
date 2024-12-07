import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.LeNet import LeNet
from Utils.getData import Data

def main():
    # Variabel utama
    b_size = 4
    epc = 20
    lr = 0.001
    n_cls = 6

    # Path dataset
    aug = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Augmented Images/FOLDS_AUG/"
    orig = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Original Images/FOLDS/"
    dt = Data(base_folder_aug=aug, base_folder_orig=orig)

    # Gabung dataset
    train_d = dt.dataset_train + dt.dataset_aug
    val_d = dt.dataset_valid
    t_loader = DataLoader(train_d, batch_size=b_size, shuffle=True)
    v_loader = DataLoader(val_d, batch_size=b_size, shuffle=False)

    # Inisialisasi model
    net = LeNet(num_classes=n_cls)

    # Inisialisasi fungsi loss dan optim
    loss = nn.CrossEntropyLoss()
    optimz = optim.SGD(net.parameters(), lr=lr)

    # Tracking loss
    trn_loss = []
    vld_loss = []

    for e in range(epc):
        net.train()  # Training mode
        total_loss = 0
        c_train = 0
        t_train = 0

        # Loop batch
        for x, y in t_loader:
            x = x.permute(0, 3, 1, 2).float()
            y = torch.argmax(y, dim=1)

            # Forward pass
            o = net(x)
            l = loss(o, y)

            # Backward
            optimz.zero_grad()
            l.backward()
            optimz.step()

            total_loss += l.item()
            _, pred = torch.max(o, 1)
            t_train += y.size(0)
            c_train += (pred == y).sum().item()

        # Simpan loss pelatihan
        trn_loss.append(total_loss / len(t_loader))

        # Validasi
        net.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in v_loader:
                x = x.permute(0, 3, 1, 2).float()
                y = torch.argmax(y, dim=1)

                o = net(x)
                l = loss(o, y)
                v_loss += l.item()

        vld_loss.append(v_loss / len(v_loader))

        print(f"Ep {e+1}/{epc} | Train L: {total_loss / len(t_loader):.4f} | Val L: {v_loss / len(v_loader):.4f}")

    # Save model
    torch.save(net.state_dict(), "model_final.pth")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(epc), trn_loss, "b-", label="Train")
    plt.plot(range(epc), vld_loss, "r-", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
