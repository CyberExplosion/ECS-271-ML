import pandas as pd
from sklearn.model_selection import KFold
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import torch
import torch.utils.data.sampler

class CoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        coordinates = torch.tensor(row[['x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6']].values, dtype=torch.float32)
        label = torch.tensor(row['Digit'], dtype=torch.long)
        return coordinates, label
    
class ReshapeCoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        coordinates = torch.tensor(row[['x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6']].values, dtype=torch.float32)
        label = torch.tensor(row['Digit'], dtype=torch.long)
        return coordinates.view(4, 2), label

class Loader:
    def __init__(self, dataset, batch_size, num_workers, kfoldSplit=5):
        """
        dataset: pandas dataframe
        batch_size: int
        num_workers: int
        kfoldSplit: int
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kfold = KFold(n_splits=kfoldSplit, shuffle=True, random_state=42)
        self.dataloaders = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.dataset)):
            # print(f'Number of indices by Kfold {len(train_idx)} and {len(val_idx)}')
            trainDataset = torch.utils.data.Subset(self.dataset, train_idx)
            valDataset = torch.utils.data.Subset(self.dataset, val_idx)


            train_loader = DataLoader(
                trainDataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                valDataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            # print(f'Number of data in train_loader: {len(train_loader)} and val_loader: {len(val_loader)}')
            # print(f'Length of dataset in train_loader: {len(train_loader.dataset)} and val_loader: {len(val_loader.dataset)}')
            self.dataloaders.append((fold, (train_loader, val_loader)))

    def get_data_loaders(self):
        return self.dataloaders


# Test the loader
if __name__ == "__main__":
#     dataset = CoordinateDataset("data/studentsdigits-train.csv")
#     loader = Loader(dataset, 100, 1).get_data_loaders()
# 
#     for fold, (train_loader, val_loader) in loader:
#         print(f"Fold: {fold}")
#         print(f"Train loader: {train_loader}")
#         print(f"Val loader: {val_loader}")
#         print(f"Train loader length: {len(train_loader)}")
#         print(f"Val loader length: {len(val_loader)}")
#         train_loader
# 
#         for coordinates, labels in train_loader:
#             print(f"Coordinates: {coordinates}")
#             print(f"Labels: {labels}")
# 
#         break
    reshaped = ReshapeCoordinateDataset("data/studentsdigits-modified.csv")
    loader = Loader(reshaped, 100, 1).get_data_loaders()
    
    for fold, (train_loader, val_loader) in loader:
        # print(f"Fold: {fold}")
        # print(f"Train loader: {train_loader}")
        # print(f"Val loader: {val_loader}")
        # print(f"Train loader length: {len(train_loader)}")
        # print(f"Val loader length: {len(val_loader)}")
        # train_loader

        for coordinates, labels in train_loader:
            print(f"Coordinates: {coordinates.shape}")
            print(f"Labels: {labels.shape}")
            break

        break
