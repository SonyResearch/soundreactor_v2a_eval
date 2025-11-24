import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models
import torchaudio
import librosa

import sys
sys.path.append('..')
from utils import sourcesep
from config import params
import models

# -------------------------------------------------------------------------------------- #

class AudioEncoder(nn.Module):
    # base encoder
    def __init__(self, args, pr, device=None):
        super(AudioEncoder, self).__init__()
        self.pr = pr
        self.args = args
        self.num_classes = pr.num_classes
        self.wav2spec = args.wav2spec
        self.trans = self.stft_transform()

    def stft_transform(self):
        win_length = 256
        n_fft = self.pr.n_fft

        if self.pr.clip_length in [0.96, 2.55]:
            hop_length = 160
        else:
            sample_num = int(self.pr.clip_length * self.pr.samp_sr)
            if self.args.finer_hop:
                hop_length = int(sample_num // 256)
            else:
                hop_length = int(sample_num // 128)

        trans = torchaudio.transforms.Spectrogram(
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            power=None  # Returns a complex tensor
        )
        return trans

    def unfold2patch(self, audio):
        '''
            audio shape: (N, 1, L)
        '''
        patch_size = int(self.pr.clip_length * self.pr.samp_sr)
        # audio shape: (N, 1, patch_num, patch_size)
        audio = audio.unfold(-1, patch_size, self.pr.patch_stride)
        # audio shape: (N, patch_num, 1, patch_size)
        audio = audio.permute(0, 2, 1, 3)
        return audio

    def wav2stft(self, audio):
        '''
            waveform shape: (N, K, 1, L)
        '''
        audio = audio.squeeze(-2)
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ FIX for modern torchaudio ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # Modern torchaudio returns a 4D complex tensor. Convert it to a 5D real tensor
        # to match the old behavior (real/imaginary parts as the last dimension).
        spec = torch.view_as_real(self.trans(audio))
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ FIX for modern torchaudio ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        # Now the permute for a 5D tensor will work correctly.
        spec = spec.permute(0, 1, 4, 3, 2)[..., :-1, :-1]
        return spec

# -------------------------------------------------------------------------------------- #

class WaveNet(AudioEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(WaveNet, self).__init__(args, pr, device)
        backbone = args.backbone if backbone is None else backbone
        self.wav2spec = args.wav2spec

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ FIX for modern torchvision ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        if backbone in ['resnet9']:
            in_channels = 2
            # Use the official ResNet class for custom architectures
            model = torchvision.models.resnet.ResNet(
                block=torchvision.models.resnet.BasicBlock,
                layers=[1, 1, 1, 1],
                num_classes=pr.feat_dim  # Set the final output dimension directly
            )
            # Replace the first convolutional layer for 2-channel input
            model.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.net = model

        elif backbone in ['resnet18']:
            in_channels = 2
            # Use the official resnet18 function with weights=None for random initialization
            model = torchvision.models.resnet.resnet18(weights=None)

            # Replace the first convolutional layer
            model.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            # Replace the final fully-connected layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, pr.feat_dim)
            self.net = model
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ FIX for modern torchvision ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.criterion = models.__dict__[pr.loss](args, pr, device)

    def forward(self, inputs, evaluate=False, loss=False):
        '''
            audio_left: (N, K, C, D) or (N, K, C, H, W)
            audio_right: (N, K, C, D) or (N, K, C, H, W)
        '''
        audio_left = inputs['left_audios']
        audio_right = inputs['right_audios']
        delay_time = inputs['delay_time']

        audio_left = self.encode_audio(audio_left)
        audio_right = self.encode_audio(audio_right)

        output = {
            'audio_left': audio_left,
            'audio_right': audio_right,
            'delay_time': delay_time
        }

        if evaluate:
            res = self.criterion.evaluate(output)
            return res
        if loss:
            loss = self.criterion(output).view(1, -1)
            return loss

        return output

    def encode_audio(self, audio):
        audio = self.unfold2patch(audio)
        if self.args.wav2spec:
            audio = self.wav2stft(audio)
        audio_size = audio.shape
        audio = audio.contiguous().view(audio_size[0] * audio_size[1], *audio_size[2:])
        audio = self.net(audio)
        audio = audio.contiguous().view(*audio_size[:2], -1)
        return audio

# The rest of the classes (WaveAugNet, WaveMixNet) inherit from WaveNet and do not need changes.
class WaveAugNet(WaveNet):
    pass
class WaveMixNet(WaveNet):
    pass