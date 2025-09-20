# engine/policy_value_net.py
import torch, torch.nn as nn, torch.nn.functional as F

class PVNetPolicyValue(nn.Module):
    """
    Simple conv trunk -> shared feat -> heads:
      - policy: 4096 logits mapping to (from,to) pairs
      - value : tanh scalar
    """
    def __init__(self, ch=96, scalar_dim=5):
        super().__init__()
        self.c1 = nn.Conv2d(12, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b3 = nn.BatchNorm2d(ch)

        self.sc = nn.Linear(scalar_dim, ch)

        self.trunk_fc = nn.Linear(ch * 8 * 8, 512)
        # policy: 64*64 logits (from,to)
        self.pi = nn.Linear(512, 64 * 64)
        # value head
        self.v1 = nn.Linear(512, 128)
        self.v2 = nn.Linear(128, 1)

    def forward(self, planes, scalars):
        x = F.relu(self.b1(self.c1(planes)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        s = F.relu(self.sc(scalars)).unsqueeze(-1).unsqueeze(-1)
        x = x + s.expand_as(x)
        x = x.flatten(1)
        x = F.relu(self.trunk_fc(x))
        # policy logits (no softmax here)
        logits = self.pi(x)                  # (B,4096)
        v = torch.tanh(self.v2(F.relu(self.v1(x))))  # (B,1)
        return logits, v
