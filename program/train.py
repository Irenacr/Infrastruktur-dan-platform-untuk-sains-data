import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.LeNet import LeNet 
from Utils.getData import Data  

def main():
    BATCH_SIZE = 4
    EPOCH = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    
    aug_path = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/Ideapad slim 3/Documents/infrastruktur data/tugas_infrasntruktur_teori/Tugas/dataset/Original Images/FOLDS/"
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    train_data = dataset.dataset_train + dataset.dataset_aug
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # LeNet models
    model = LeNet(num_classes=NUM_CLASSES)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    for epoch in range(EPOCH):
        loss_train = 0.0
        correct_train = 0
        total_train = 0
        model.train()

        for batch_idx, (src, trg) in enumerate(train_loader):
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

        accuracy_train = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{EPOCH}], Train Loss: {loss_train / len(train_loader):.4f}, Accuracy: {accuracy_train:.2f}%")

        train_losses.append(loss_train / len(train_loader))
    torch.save(model.state_dict(), "trained_model4.pth")

    # Plot the training loss
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training.png")
    

if __name__ == "__main__":
    main()
