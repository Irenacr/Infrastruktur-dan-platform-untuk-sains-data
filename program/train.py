import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.simplecnn import simplecnn
from Utils.getData import Data

def main():
    BATCH_SIZE = 4
    EPOCH = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6

    # Paths to dataset
    aug_path = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Original Images/FOLDS/"
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    train_data = dataset.dataset_train + dataset.dataset_aug
    val_data = dataset.dataset_valid  # Tambahkan data validasi
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # LeNet model
    model = simplecnn(num_classes=NUM_CLASSES)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCH):
        # Training loop
        model.train()
        loss_train = 0.0
        correct_train = 0
        total_train = 0

        for src, trg in train_loader:
            src = src.permute(0, 3, 1, 2).float()  
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        train_losses.append(loss_train / len(train_loader))

        # Validation loop
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for src, trg in val_loader:
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)
                loss_val += loss.item()

        val_losses.append(loss_val / len(val_loader))

        print(f"Epoch [{epoch + 1}/{EPOCH}] | Train Loss: {loss_train / len(train_loader):.4f} | Val Loss: {loss_val / len(val_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "trained_model4.pth")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label="Training Loss")
    plt.plot(range(EPOCH), val_losses, color="#ff6666", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("training.png")
    plt.show()

if __name__ == "__main__":
    main()
