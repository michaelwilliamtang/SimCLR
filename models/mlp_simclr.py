import torch.nn as nn

from exceptions.exceptions import InvalidBackboneError


class MLPSimCLR(nn.Module):

    def __init__(self, in_dim, hid_dim=1000, proj_dim=1000, out_dim=1000):
        super(MLPSimCLR, self).__init__()

        # trained backbone is depth-4 MLP, with a depth-1 MLP projection head
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.backbone(x)
