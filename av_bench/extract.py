import logging
from pathlib import Path
import os

import torch
import torchaudio
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List
import pandas as pd


from av_bench.data.audio_dataset import (AudioDataset, ImageBindAudioDataset,
                                         SynchformerAudioDataset, pad_or_truncate)
from av_bench.extraction_models import ExtractionModels
from av_bench.synchformer.synchformer import Synchformer

log = logging.getLogger()


def encode_audio_with_sync(synchformer: Synchformer, x: torch.Tensor,
                           mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    b, t = x.shape

    # partition the video
    segment_size = 10240
    step_size = 10240 // 2
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    x = mel(x)
    x = torch.log(x + 1e-6)
    x = pad_or_truncate(x, 66)

    mean = -4.2677393
    std = 4.5689974
    x = (x - mean) / (2 * std)
    # x: B * S * 128 * 66
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x


@torch.inference_mode()
def extract(
    audio_path: Path,
    output_path: Path,
    *,
    clap_model_path: Path,
    syncformer_ckpt_path: Path,
    start_time: Optional[float] = 0.0,
    csv_path: Optional[Path] = None, 
    audio_length: float = 8.0,
    batch_size: int = 128,
    num_workers: int = 32,
    device: str,
    skip_video_related: bool = False,
    mono_type: str = "mean"
    ):
    
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        col_candidates = [c for c in df.columns if c.lower() in ("filepath")]
        if not col_candidates:
            raise ValueError("No suitable column found in CSV for video paths.")
        video_col = col_candidates[0]
        audio_paths: List[Path] = []
        for v in df[video_col].astype(str):
            apath = Path(v)
            if apath.is_file():
                audio_paths.append(apath)
            else:
                log.warning(f"[SKIP] audio not found for CSV entry: {apath}")
        audios = sorted(audio_paths, key=lambda x: x.stem)

    else:
        audio_paths = []
        for root, dirs, files in os.walk(audio_path):
            for fname in files:
                if fname.lower().endswith(('.wav', '.flac', '.mp4', '.mp3')):
                    audio_paths.append(Path(root) / fname)
        audios = sorted(audio_paths, key=lambda x: x.stem)
    log.info(f'{len(audios)} audio/video files found (recursively).')

    models = ExtractionModels(clap_model_path, syncformer_ckpt_path).to(device).eval()
    dataset = AudioDataset(audios, audio_length=audio_length, start_time=start_time, sr=16000, mono=True, mono_type=mono_type)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # extract features with VGGish
    out_dict = {}
    for wav, filename in tqdm(loader, desc="VGGish"):
        wav = wav.squeeze(1).float()
        features = models.vggish(wav).cpu()
        for i, f_name in enumerate(filename):
            out_dict[f_name] = features[i]

    output_path.mkdir(parents=True, exist_ok=True)
    vggish_feature_path = output_path / f'vggish_features_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_dict)} features to {vggish_feature_path}')
    torch.save(out_dict, vggish_feature_path)
    del out_dict
    
    
    
    dataset = AudioDataset(audios, start_time=start_time, audio_length=audio_length, sr=32_000, mono=True, mono_type=mono_type)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # extract features with PaSST
    out_features = {}
    out_logits = {}
    for wav, filename in tqdm(loader, desc="PaSST"):
        wav = wav.to(device)
        wav = wav.squeeze(1).float()
        if (wav.size(-1) >= 320000):
            wav = wav[..., :320000]
        else:
            wav = torch.nn.functional.pad(wav, (0, 320000 - wav.size(-1)))

        features = models.passt_model(wav).cpu()
        # see https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/wrapper.py#L40
        # logits is 527 dim; features is 768
        logits = features[:, :527]
        features = features[:, 527:]
        for i, f_name in enumerate(filename):
            out_features[f_name] = features[i]
            out_logits[f_name] = logits[i]
    output_path.mkdir(parents=True, exist_ok=True)
    passt_feature_path = output_path / f'passt_features_embed_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_features)} features to {passt_feature_path}')
    torch.save(out_features, passt_feature_path)

    passt_feature_path = output_path / f'passt_logits_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_logits)} features to {passt_feature_path}')
    torch.save(out_logits, passt_feature_path)
    del out_features
    del out_logits
    
    # extract features with PANNs
    out_dict = {}
    for wav, filename in tqdm(loader, desc="PANNs"):
        wav = wav.to(device)
        original_length = wav.shape[-1]
        wav = wav.squeeze(1).float()

        features = models.panns(wav)
        features = {k: v.cpu() for k, v in features.items()}
        for i, f_name in enumerate(filename):
            out_dict[f_name] = {k: v[i] for k, v in features.items()}

    output_path.mkdir(parents=True, exist_ok=True)
    pann_feature_path = output_path / f'pann_features_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_dict)} features to {pann_feature_path}')
    torch.save(out_dict, pann_feature_path)
    del out_dict


    dataset = AudioDataset(audios, audio_length=audio_length, start_time=start_time, sr=48_000, mono=True, mono_type=mono_type)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # extract features with OpenL3
    out_dict = {}
    for wav, filename in tqdm(loader, desc="OpenL3"):
        wav = wav.to(device)         # (B,2,N)
        wav = wav[:, 0, :]     
        emb, _ = models.openl3.get_audio_embedding(
            wav, sr=48000,
            content_type="env",
            input_repr="mel256",
            embedding_size=512,
            hop_size=0.5,
        )
        openl3_features = emb.cpu()            # (B, T, 512)
        for i, f_name in enumerate(filename):
            out_dict[f_name] = openl3_features[i]   # store (T,1024)

    output_path.mkdir(parents=True, exist_ok=True)
    openl3_features_path = output_path / f'openl3_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_dict)} features to {openl3_features_path}')
    torch.save(out_dict, openl3_features_path)
    del out_dict
    
    # extract features with LAION-CLAP
    out_dict = {}
    for wav, filename in tqdm(loader, desc="L-CLAP"):
        wav = wav.to(device)         # (B,2,N)
        wav = wav[:, 0, :]            # (B,N)

        clap_features = models.laion_clap.get_audio_embedding_from_data(
            wav, use_tensor=True
        )  # (B, D)
        for i, f_name in enumerate(filename):
            out_dict[f_name] = clap_features[i]  # store (2D,)
    clap_feature_path = output_path / f'clap_laion_audio_mono_{mono_type}.pth'
    log.info(f'Saving {len(out_dict)} features to {clap_feature_path}')
    torch.save(out_dict, clap_feature_path)
    del out_dict
    
    if not skip_video_related:
        dataset = ImageBindAudioDataset(audios, start_time=start_time, audio_length=audio_length, mono_type=mono_type)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader, desc="ImageBind"):
            wav = wav.squeeze(1).to(device)
            features = models.imagebind({ModalityType.AUDIO: wav})[ModalityType.AUDIO].cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        output_path.mkdir(parents=True, exist_ok=True)
        imagebind_feature_path = output_path / f'imagebind_audio_{mono_type}.pth'
        log.info(f'Saving {len(out_dict)} features to {imagebind_feature_path}')
        torch.save(out_dict, imagebind_feature_path)

        # extract features with Synchformer
        dataset = SynchformerAudioDataset(audios, start_time=start_time, duration=audio_length, mono_type=mono_type)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader, desc="Synchformer"):
            wav = wav.to(device)
            features = encode_audio_with_sync(
                models.synchformer, wav,
                models.sync_mel_spectrogram
            ).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        output_path.mkdir(parents=True, exist_ok=True)
        synchformer_feature_path = output_path / f'synchformer_audio_{mono_type}.pth'
        log.info(f'Saving {len(out_dict)} features to {synchformer_feature_path}')
        torch.save(out_dict, synchformer_feature_path)