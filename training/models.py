import torch, torch.nn as nn, torch.nn.functional as F

class PVNetCNN(nn.Module):
    def __init__(self, ch=96, scalar_dim=5):
        super().__init__()
        self.c1 = nn.Conv2d(12, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b3 = nn.BatchNorm2d(ch)
        self.sc = nn.Linear(scalar_dim, ch)
        self.v1 = nn.Linear(ch * 8 * 8, 256)
        self.v2 = nn.Linear(256, 1)

    def forward(self, planes, scalars):
        x = F.relu(self.b1(self.c1(planes)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        # fuse scalars
        s = F.relu(self.sc(scalars))
        s = s.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = x + s
        x = x.flatten(1)
        x = F.relu(self.v1(x))
        v = torch.tanh(self.v2(x))
        return v

class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, scalar_dim=5):
        super().__init__()
        self.embed = nn.Linear(12, d_model)  # per-square 12-dim -> d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sc = nn.Linear(scalar_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, planes, scalars):
        # planes: (B,12,8,8) -> (B,64,12)
        B = planes.size(0)
        x = planes.view(B, 12, 64).transpose(1, 2)  # (B,64,12)
        x = self.embed(x)                            # (B,64,dm)
        # add scalar bias
        s = torch.relu(self.sc(scalars)).unsqueeze(1)  # (B,1,dm)
        x = x + s
        x = self.enc(x)                              # (B,64,dm)
        x = x.reshape(B, -1)
        v = self.head(x)                             # (B,1) in [-1,1]
        return v
