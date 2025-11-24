import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

log = logging.getLogger()


_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VGGSound(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = 'sets/vgg3-train.tsv',
        duration_sec: float = 8.0,
    ):
        self.root = Path(root)

        videos = sorted(os.listdir(self.root))
        videos = set([Path(v).stem for v in videos])  # remove extensions
        self.labels = {}
        self.videos = []
        missing_videos = []

        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep='\t', dtype={'id': str}).to_dict('records')
        for record in df_list:
            id = record['id']
            label = record['label']
            if id in videos:
                self.labels[id] = label
                self.videos.append(id)
            else:
                missing_videos.append(id)

        # if local_rank == 0:
        #     log.info(f'{len(videos)} videos found in {root}')
        #     log.info(f'{len(self.videos)} videos found in {tsv_path}')
        #     log.info(f'{len(missing_videos)} videos missing in {root}')
        self.duration_sec = duration_sec
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)
        
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.resampler = {}

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        label = self.labels[video_id]

        reader = StreamingMediaDecoder(self.root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        sync_chunk = data_chunk[1]
        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f'Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            )
        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_id}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'id': video_id,
            'caption': label,
            'sync_video': sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.labels)