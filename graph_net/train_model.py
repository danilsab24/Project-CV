import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch import optim
from point_net import PointNet
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Percorso del dataset
dataset_path = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\npy_dataset"

# Dizionario per la mappatura dei caratteri agli interi
char2int = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12,
    "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25,
    "nothing": 26, "del": 27, "space": 28
}

class PointsDataSet(Dataset):
    def __init__(self, path, items):
        self.path = path
        self.items = items

    def __getitem__(self, item):
        label, file_name = self.items[item]
        points = np.load(os.path.join(self.path, label, file_name))
        points = np.array(points)

        if label not in char2int:
            raise ValueError(f"Label '{label}' not found in char2int dictionary.")
        
        return points, char2int[label]

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))

        ims = torch.stack([torch.Tensor(im) for im in ims]).float().to(device)
        ce_masks = torch.tensor(classes).long().to(device)

        return ims, ce_masks


def split_files(path, divs):
    if divs[0] + divs[1] + divs[2] != 1.0:
        print(f'Wrong divisions: Train={divs[0]} Validation={divs[1]} Test={divs[2]} Total={divs[0] + divs[1] + divs[2]}')
        sys.exit()

    items = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            items.append((os.path.basename(root), file))

    np.random.shuffle(items)
    size = len(items)
    train = int(size * divs[0])
    val = int(size * divs[1])
    test = int(size * divs[2])
    total = train + test + val
    train = train + (size - total)

    # train - validation - test
    return items[:train], items[train:train+val], items[train+val:]


def train_batch(model, data, optmz, loss_fn):
    model.train()
    points, classes = data
    _masks = model(points)
    optmz.zero_grad()
    loss, acc = loss_fn(_masks, classes)
    loss.backward()
    optmz.step()

    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)

    return loss.item(), acc.item()


def pointnet_loss(preds, targets):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc


def main():
    # Impostazioni principali
    model_path = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\model"
    results_path = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\results"
    model_name = "point_net_model.pth"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    train_percent = 0.7
    validation_percent = 0.15
    test_percent = 0.15
    train_files, val_files, _ = split_files(dataset_path, (train_percent, validation_percent, test_percent))

    n_epochs = 5
    batch_size = 16
    learning_rate = 0.001
    optimizer_name = "Adam"

    train_ds = PointsDataSet(dataset_path, train_files)
    val_ds = PointsDataSet(dataset_path, val_files)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=val_ds.collate_fn, drop_last=True)

    model = PointNet(len(char2int)).to(device)
    loss_function = pointnet_loss
    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0, threshold=0.001, min_lr=1e-5, threshold_mode='abs')

    loss_train_all = []
    acc_train_all = []
    loss_val_all = []
    acc_val_all = []

    for epoch in range(n_epochs):
        print(f'Current epoch: {epoch}')
        loss_ep = []
        acc_ep = []

        # Aggiungi tqdm per la barra di progresso durante il training
        for points, classes in tqdm(train_dl, desc="Training", leave=False):
            loss, acc = train_batch(model, (points, classes), optimizer, loss_function)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_train_all.append(np.mean(loss_ep))
        acc_train_all.append(np.mean(acc_ep))

        loss_ep = []
        acc_ep = []

        # Aggiungi tqdm per la barra di progresso durante la validazione
        for points, classes in tqdm(val_dl, desc="Validation", leave=False):
            loss, acc = validate_batch(model, (points, classes), loss_function)
            loss_ep.append(loss)
            acc_ep.append(acc)
        loss_val_all.append(np.mean(loss_ep))
        acc_val_all.append(np.mean(acc_ep))

        val_loss = np.mean(loss_ep)
        scheduler.step(val_loss)

        print(f'Loss train {loss_train_all[-1]} Loss val {loss_val_all[-1]} Acc train {acc_train_all[-1]} Acc val {acc_val_all[-1]}')
        with open(os.path.join(results_path, 'status.csv'), 'a') as f:
            f.write(f'{loss_train_all[-1]}; {loss_val_all[-1]}; {acc_train_all[-1]}; {acc_val_all[-1]}\n')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_path, model_name))

    torch.save(model.state_dict(), os.path.join(model_path, model_name))

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(loss_train_all, color='green', label='Loss Train')
    axs[0].plot(loss_val_all, color='red', label='Loss Val')
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(acc_train_all, color='green', label='Acc Train')
    axs[1].plot(acc_val_all, color='red', label='Acc Val')
    axs[1].set_title("Accuracy")
    axs[1].legend()

    plt.savefig(os.path.join(results_path, f'metrics_point_net_{n_epochs}epochs_{learning_rate}lr.png'))
    plt.show()


if __name__ == '__main__':
    main()
