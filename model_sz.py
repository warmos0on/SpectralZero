import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dataset):
        super().__init__()
        if dataset == "Houston":
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
            bias = True
        elif dataset == "Indian":
            kernel_size = 3
            padding = 1
            bias = True
        elif dataset == "LongKou":
            kernel_size = 3
            padding = 1
            bias = False
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channel))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channel))
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channel))

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1 + x3, inplace=True)
        return out


class SpectralEx(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, patch_size, n_bands, dim, dataset=None):
        super(SpectralEx, self).__init__()
        self.n_bands = n_bands
        self.block1 = ResidualBlock(in_channel, out_channel1, dataset)
        
        # Houston 
        if dataset == "Houston":
            self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1), stride=(1, 2, 2))
        elif dataset == "Indian" or dataset == "LongKou":
        # Indian Pines and LongKou
            self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        else:
            raise ValueError("Wrong Datasets!")
        
        self.block2 = ResidualBlock(out_channel1, out_channel2, dataset)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1), stride=(1, 2, 2))
        
        if dataset == "Houston":
            self.conv1 = nn.Conv3d(in_channels=out_channel2, out_channels=16, kernel_size=(3, 1, 1), bias=False)
        elif dataset == "Indian":
        # Indian Pines
            self.conv1 = nn.Conv3d(in_channels=out_channel2, out_channels=16, kernel_size=(1, 3, 3), bias=False)
        # LongKou
        elif dataset == "LongKou":
            self.conv1 = nn.Conv3d(in_channels=out_channel2, out_channels=8, kernel_size=(1, 3, 3), bias=False)
        self.patch_size = patch_size

        
        self.spectral_lower = nn.Sequential(
            nn.Conv3d(in_channels=out_channel2, out_channels=1, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(1)
            )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        _, final_dim = self._get_spatial_size()
        _, spectral_final_dim = self._get_spectral_size()
        
        if dataset == "Houston":
        # Houston
            self.spectral_block = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm1d(16),
                nn.ReLU()
            )
        # Indian Pines
        elif dataset == "Indian":
            self.spectral_block = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm1d(4),
                nn.ReLU()
            )
        elif dataset == "LongKou":
            self.spectral_block = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm1d(8),
                nn.ReLU()
            )
        self.mlp = nn.Sequential(nn.Linear(in_features=final_dim, out_features=dim, bias=True), nn.BatchNorm1d(dim), nn.ReLU())
        if dataset == "Houston":
            self.spectral_mlp = nn.Sequential(nn.Linear(in_features=spectral_final_dim * 16, out_features=dim, bias=True), nn.BatchNorm1d(dim), nn.ReLU())
        if dataset == "LongKou":
            self.spectral_mlp = nn.Sequential(nn.Linear(in_features=spectral_final_dim * 8, out_features=dim, bias=True), nn.BatchNorm1d(dim), nn.ReLU())
        if dataset == "Indian":
            self.spectral_mlp = nn.Sequential(nn.Linear(in_features=spectral_final_dim * 4, out_features=dim, bias=True), nn.BatchNorm1d(dim), nn.ReLU())
        
    def _get_spatial_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.n_bands, self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            final_patch = x.shape[-1]
            x = self.conv1(x)
            x = x.view(x.shape[0], -1)
            s = x.size()[1]
        return final_patch, s
    
    def _get_spectral_size(self):
        with torch.no_grad():
            x = torch.zeros((2, 1, self.n_bands, self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            final_patch = x.shape[-1]
            x_spectral = self.spectral_lower(x)
            x_spectral = torch.squeeze(x_spectral)
            x_spectral = self.global_pool(x_spectral)
            x_spectral = torch.squeeze(x_spectral).unsqueeze(1)
            x_spectral = x_spectral.view((x_spectral.shape[0], -1))
            s = x_spectral.size()[1]
        return final_patch, s

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        features = self.conv1(x)
        spatital_features = features.view(features.shape[0], -1)
        spatital_features = self.mlp(spatital_features)
        spectral_feature = self.spectral_lower(x)
        spectral_feature = torch.squeeze(spectral_feature)
        spectral_feature = self.global_pool(spectral_feature)
        spectral_feature = torch.squeeze(spectral_feature).unsqueeze(1)
        spectral_feature = self.spectral_block(spectral_feature)
        spectral_feature = spectral_feature.view((spectral_feature.shape[0], -1))
        spectral_feature = self.spectral_mlp(spectral_feature)
        return spatital_features, spectral_feature


class FinalOut(nn.Module):
    def __init__(self, dim, embedding_dim, num_classes, spectral_dim):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.projection = nn.Linear(dim, embedding_dim, bias=False)
        self.spectral_projection = nn.Linear(spectral_dim, embedding_dim, bias=False)

    def forward(self, spatital_feature, spectral_feature, classication=True):
        cls = 0
        projection = self.projection(spatital_feature)
        spectral_projection = self.spectral_projection(spectral_feature)
        if classication:
            cls = self.classifier(spatital_feature)
        return projection, spectral_projection, cls


class SpectralZero(nn.Module):
    def __init__(self, hyper) -> None:
        super().__init__()
        vision_patch_size = hyper["patch_size"]
        embed_dim = hyper["embed_dim"]
        num_classes = hyper["seen_class"]
        inchannel = hyper["bands"]
        self.device = hyper["device"]
        dataset = hyper["dataset"]

        self.spectral_ratio = hyper["spectral_ratio"]
        self.visual = SpectralEx(1, 8, 16, vision_patch_size, inchannel, hyper["vision_dim"], dataset)
        self.final_pro = FinalOut(hyper["vision_dim"], embed_dim, num_classes, hyper["spectral_dim"])

        self.ce_loss = nn.CrossEntropyLoss()
        if dataset == "Houston" or dataset == "Indian":
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        elif dataset == "LongKou":
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.17))
        self.hyperparams = hyper

    @property
    def dtype(self):
        return torch.float32

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def forward(self, image, text_project: torch.Tensor, label):
        device = self.device
        spatital_feature, spectral_feature = self.encode_image(image)
        spatital_project, spectral_project, cls_head = self.final_pro.forward(spatital_feature, spectral_feature, classication=True)

        spatital_project = spatital_project / spatital_project.norm(dim=1, keepdim=True)
        spectral_project = spectral_project / spectral_project.norm(dim=1, keepdim=True)
        
        text_project = text_project / text_project.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        spatital_cossim = spatital_project @ text_project.t()
        spectral_cossim = spectral_project @ text_project.t()
        cosine_similar = logit_scale * ((1 - self.spectral_ratio) * spatital_cossim + self.spectral_ratio * spectral_cossim)

        if self.training:
            cls_loss = self.ce_loss(cls_head, label.long())
            clip_label = torch.arange(cosine_similar.size(0)).long().to(device)
            clip_loss = self.ce_loss(cosine_similar, clip_label)
            return cls_loss, clip_loss, cls_head
        else:
            _, pred = torch.max(F.softmax(cosine_similar, dim=1), dim=1)
            return pred, cls_head
