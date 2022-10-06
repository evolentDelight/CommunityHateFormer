import pytorch_lightning as pl
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch 
from dataset import GraphDataset

class GATLayer(nn.Module):
    def __init__(self, input_size, output_size, heads):
        self.dropout = nn.Dropout(p=0.6)
        self.conv = GATConv(input_size, output_size, heads)
        
    def forward(self, x, edge_index):
        return self.cond(self.dropout(x), edge_index)



class GATModel(pl.LightningModule):
    def __init__(self, input_size=769, output_size=4, hidden_size=768, layer_count=4):
        layers = [GATLayer(input_size, hidden_size // 8, 8)]
        for i in range(2, layer_count):
            layers += [GATLayer(hidden_size, hidden_size // 8, 8)]
        self.output_layer = GATLayer(hidden_size, output_size, 8)

        self.modules = nn.ModuleList(layers)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index

        for layer in self.modules:
            x = layer(x, edge_index)
            x = nn.ELU()(x)

        x = self.output_layer(x)
        x = nn.Sigmoid()(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=0.005, weight_decay=5e-4)
        
        return optimizer
    
    def training_step(self, batched_data, batch_idx):
        y_pred = self.forward(batched_data)
        y_true = batched_data.y.flatten(-1)
        mask = batched_data.mask.flatten(-1)

        y_pred, y_true = y_pred[mask], y_true[mask]

        loss = self.loss_fn(y_pred, y_true)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batched_data, batch_idx):
        y_pred = self.forward(batched_data)
        y_true = batched_data.y.flatten(-1)
        mask = batched_data.mask.flatten(-1)

        y_pred, y_true = y_pred[mask], y_true[mask]

        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss', loss}
    
    def validation_epoch_end(self, outputs):
        losses = torch.cat([loss['val_loss'].flatten() for loss in outputs if ~torch.isnan(loss['val_loss'])]).mean()
        
        try:
            self.log('valid_cross_entopy', losses)
        except:
            pass
    
    def test_step(self, batched_data, batched_idx):
        mask = batched_data.mask.flatten(-1)
        #y_true = batched_data.y.flatten(end_dim=1)
        #print(mask.shape)
        y_true = batched_data.y[mask]
        y_pred = self(batched_data)
        y_pred = y_pred[mask]

        loss = self.loss_fn(y_pred, y_true)
        return {
            'test_loss': loss 
        }

    def test_epoch_end(self, outputs):
        losses = torch.cat([loss['test_loss'].flatten() for loss in outputs if ~torch.isnan(loss['test_loss'])]).mean()
        self.log('test_cross_entropy', losses)



if __name__ == "__main__":
    model = GATModel()
    print('total params:', sum(p.numel() for p in model.parameters()))
    dataset = GraphDataset()
    trainer = pl.Trainer(checkpoint_callback=True, batch_size=8, precision=32, max_epochs=10)

    result = trainer.fit(model, datamodule=dataset)
        


        