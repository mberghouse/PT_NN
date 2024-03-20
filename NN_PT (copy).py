import torch
import torch.nn as nn
import gc
import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, num_particles, sequence_length, d_model=384, nhead=16, num_encoder_layers=12):
        super(TemporalTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Initial convolution layer to extract features from the image
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        # Flatten and linear layer to match the transformer's input dimension
        self.fc = nn.Linear(256 * 80 * 80, d_model)  # Adjust based on conv output
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoders for x and y
        self.decoder_x = nn.Linear(d_model, num_particles)
        self.decoder_y = nn.Linear(d_model, num_particles)
    
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        assert seq_length == self.sequence_length, "Input sequence length doesn't match the defined sequence length"

        # Process each image in the sequence
        embeddings = []
        for t in range(seq_length):
            features = self.conv(x[:, t])
            embeddings.append(self.fc(features.view(batch_size, -1)))
        
        # Shape for transformer: S x N x E (sequence length x batch size x embedding dim)
        embeddings = torch.stack(embeddings).permute(1, 0, 2)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer_encoder(embeddings)
        
        # Decode the transformer outputs for all images in the sequence
        out_x = self.decoder_x(transformer_out)
        out_y = self.decoder_y(transformer_out)
        
        return out_x, out_y

# Example usage


class ParticleTracker(nn.Module):
    def __init__(self, num_particles):
        super(ParticleTracker, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        # ConvLSTM
        self.convlstm = nn.LSTMCell(256*90*90, 64)  # Adjust based on encoder's output
        
        # Decoder
        self.decoder_x = nn.Linear(64, num_particles)
        self.decoder_y = nn.Linear(64, num_particles)
    
    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        hidden, cell = None, None
        outputs_x, outputs_y = [], []

        for t in range(seq_length):
            # Encoder
            features = self.encoder(x[:, t])
            features = features.view(batch_size, -1)

            # ConvLSTM
            if hidden is None or cell is None:
                hidden, cell = self.convlstm(features)
            else:
                hidden, cell = self.convlstm(features, (hidden, cell))

            # Decoder
            out_x = self.decoder_x(hidden)
            out_y = self.decoder_y(hidden)
            outputs_x.append(out_x)
            outputs_y.append(out_y)

        outputs_x = torch.stack(outputs_x, dim=1)
        outputs_y = torch.stack(outputs_y, dim=1)

        return outputs_x, outputs_y
        
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class ParticleDataset(Dataset):
    def __init__(self, image_paths, x_gt, y_gt, sequence_length=30):
        self.image_paths = image_paths
        self.x_gt = x_gt
        self.y_gt = y_gt
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.image_paths) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq_images = []
        for i in range(self.sequence_length):
            img = cv2.imread(self.image_paths[idx + i])
            img = cv2.resize(img, (320, 320))
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)/255
            seq_images.append(img)

        # Get the sequence of x and y coordinates for all particles
        #print (self.x_gt[0:10])
        #print (idx)
        x = self.x_gt[idx: idx + self.sequence_length]
        y = self.y_gt[idx: idx + self.sequence_length]

        return torch.stack(seq_images), x, y


# Example usage
top_dir='1000part_4xspeed_heterogeneous/'
xc_data = pd.read_csv('xc_1000part_4xspeed.csv', header=None)
yc_data = pd.read_csv('yc_1000part_4xspeed.csv', header=None)
counts=[]
for i in range(900):
    counts.append(yc_data.iloc[:,i].count())
max_counts = min(counts)
print (max_counts)
train_image_paths = os.listdir(top_dir)
for i in range(len(train_image_paths)):
    train_image_paths[i] = os.path.join(top_dir,'Fig_'+str(i+1)+'.jpg')
train_image_paths=train_image_paths[0:max_counts]

train_dataset = ParticleDataset(train_image_paths, np.array(xc_data)[:,0:900], np.array(yc_data)[:,0:900])
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
# def custom_collate_fn(batch):
#     images, x_gt, y_gt = zip(*batch)
    
#     # Ensure x_gt and y_gt are of consistent size
#     x_gt = [torch.tensor(x, dtype=torch.float32) if len(x) == 670 else torch.tensor(np.pad(x, (0, 670-len(x))), dtype=torch.float32) for x in x_gt]
#     y_gt = [torch.tensor(y, dtype=torch.float32) if len(y) == 670 else torch.tensor(np.pad(y, (0, 670-len(y))), dtype=torch.float32) for y in y_gt]
    
#     return torch.stack(images, 0), torch.stack(x_gt, 0), torch.stack(y_gt, 0)

train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True,shuffle=True,  drop_last=True, num_workers=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalTransformer(num_particles=900, sequence_length=30).to('cuda')
#model = ParticleTracker(num_particles=900).to('cuda')
criterion = nn.MSELoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=.004)
scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 260, T_mult=2, eta_min=1e-12, last_epoch=- 1, verbose=False)
epochs=200

for epoch in range(epochs):
    model.train()
    for images, x_gt, y_gt in train_loader:
        images = images.float().to('cuda')
        x_gt = x_gt.float().to('cuda')
        #x_gt[torch.isnan(x_gt)] = torch.mean(x_gt)
        y_gt = y_gt.float().to('cuda')
        #y_gt[torch.isnan(y_gt)] = torch.mean(y_gt)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            x_pred, y_pred = model(images)
            #x_pred = x_pred.to('cpu')
            #y_pred = y_pred.to('cpu')
            #torch.cuda.empty_cache()
            loss = criterion(x_pred, x_gt)+ criterion(y_pred, y_gt)
            scheduler.step() 
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        gc.collect()

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, LR: {lr}")


torch.save(model.state_dict(), 'weights/run1')

torch.save(model, 'model/run1')
    
