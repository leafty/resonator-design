import pytorch_lightning as pl

from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
        super().__init__()
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader