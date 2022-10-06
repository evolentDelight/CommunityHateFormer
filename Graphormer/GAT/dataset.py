import pytorch_lightning as pl
from reddit_dataset import RedditDataset
from torch_geometric.data import DataLoader

class GraphDataset(pl.LightningDataModule):
    def __init__(self, num_workers, batch_size, seed):
        self.dataset = RedditDataset('/home/l2hebert/dev/processed-graphs/')
        self.batch_size = batch_size
        self.num_workers = num_workers
  
    def train_dataloader(self):
        # be mindful of including state in here bucko
        loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return loader