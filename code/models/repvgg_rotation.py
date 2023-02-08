import torch
import torch.nn as nn


class RepVGGRotation(nn.Module):
    def __init__(self):
        super(RepVGGRotation, self).__init__()
        # load the pretrained model weight
        # these feature extraction backbone parameters are freezed
        # shouldn't be changed during rotation prediction training
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True
        )
        # feature extraction backbone
        self.feat = torch.nn.Sequential(*list(self.model.children())[:-1])
        # classification FC layer
        self.fc = list(self.model.children())[-1]
        # rotation prediction FC layer
        self.fc_rotation = nn.Linear(1280, 4)

    def forward(self, x):
        x = self.feat(x)
        # flatten the feature representation
        x = x.view(x.size(0), -1)
        return self.fc(x), self.fc_rotation(x)
