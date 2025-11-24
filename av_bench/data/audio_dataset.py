import logging
from pathlib import Path
from typing import List, Tuple  

import torch
import torchaudio
import torchvision.transforms.v2 as v2
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torio.io import StreamingMediaDecoder
from torch.utils.data import Dataset

log = logging.getLogger()

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


def int16_to_float32(x):
    """
    Convert a NumPy array of int16 values to float32.
    Parameters:
        x (numpy.ndarray): A NumPy array of int16 values.
    Returns:
        numpy.ndarray: A NumPy array of float32 values, scaled from the int16 input.
    """

    return (x / 32767.0).to(torch.float32)

def float32_to_int16(x):
    """
    Converts a NumPy array of float32 values to int16 values.
    This function clips the input array values to the range [-1.0, 1.0] and then scales them to the range of int16 
    (-32768 to 32767).
    Parameters:
        x (numpy.ndarray): A NumPy array of float32 values to be converted.
    Returns:
        numpy.ndarray: A NumPy array of int16 values.
    """

    x = torch.clip(x, min=-1., max=1.)
    return (x * 32767.).to(torch.int16)


# from ImageBind
def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints

def load_waveform(
    path: Path, 
    target_sr: int = 48000, 
    duration: float = 8.0, 
    start_time: float = 0.0,
    mono: bool = False,
    mono_type: str = "mean"
) -> Tuple[torch.Tensor,int]:
    """
    Load audio track from `path`.  If it's a video container, use MoviePy;
    otherwise torchaudio.load.
    Returns (waveform, sample_rate).
    """
    ext = path.suffix.lower()
    max_samples = int(target_sr * duration)
    if ext in VIDEO_EXTS:
        dec = StreamingMediaDecoder(path)
        dec.add_basic_audio_stream(
            frames_per_chunk=max_samples, 
            sample_rate=target_sr,
            num_channels=2,
        )
        dec.seek(start_time)
        dec.fill_buffer()
        audio_chunk = dec.pop_chunks()[0]
        if audio_chunk.shape[0] < max_samples:
            pad = max_samples - audio_chunk.shape[0]
            arr = torch.nn.functional.pad(audio_chunk, (0, 0, 0, pad), mode="constant", value=0)
        else:
            arr = audio_chunk[:max_samples]
        waveform = arr.T.float()
    else:
        # pure audio file
        info = torchaudio.info(str(path))
        sr_orig = info.sample_rate
        start_frame = int(start_time * sr_orig)
        num_frames = int(duration * sr_orig)
        
        waveform, sr_orig = torchaudio.load(
            str(path),
            frame_offset=start_frame,
            num_frames=num_frames
        )
        # waveform, sr_orig = torchaudio.load(str(path))  # (channels, N)
        
        # resample if needed
        if sr_orig != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr_orig, new_freq=target_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        # stereo handling: if mono, duplicate to stereo
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        # clip or pad
        if waveform.size(1) < max_samples:
            pad = max_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :max_samples]
    
        
    sr = target_sr
    if mono:
        if mono_type == "mean":
            waveform = waveform.mean(dim=0, keepdim=True)
        elif mono_type == "left":
            waveform = waveform[0, :].unsqueeze(0)  # keep as 2D tensor
        elif mono_type == "right":
            waveform = waveform[1, :].unsqueeze(0) # keep as 2D tensor
        elif mono_type == "side":
            waveform = 0.5 * (waveform[0, :] - waveform[1, :]).unsqueeze(0)
        
    # waveform = torch.from_numpy(pyln.normalize.peak(waveform.numpy(), -1.0))
    waveform = int16_to_float32(float32_to_int16(waveform)).float()
        
    return waveform, sr

# from ImageBind
def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


# from synchformer
def pad_or_truncate(audio: torch.Tensor,
                    max_spec_t: int,
                    pad_mode: str = 'constant',
                    pad_value: float = 0.0):
    difference = max_spec_t - audio.shape[-1]  # safe for batched input
    # pad or truncate, depending on difference
    if difference > 0:
        # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
        pad_dims = (0, difference)
        audio = torch.nn.functional.pad(audio, pad_dims, pad_mode, pad_value)
    elif difference < 0:
        log.warning(f'Truncating spec ({audio.shape}) to max_spec_t ({max_spec_t}).')
        audio = audio[..., :max_spec_t]  # safe for batched input
    return audio


def pad_short_audio(audio, min_samples=32000):
    if (audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.size(-1)),
                                        mode='constant',
                                        value=0.0)
    return audio


class AudioDataset(Dataset):
    def __init__(
        self,
        datalist: List[Path],
        audio_length: float = 8.0,
        sr: int = 48000,
        start_time: float = 0.0,
        # target_lufs: float = -16.0,
        limit_num: int = None,
        mono: bool = False,
        mono_type: str = "mean"
    ):
        self.datalist = datalist[:limit_num] if limit_num else datalist
        self.sr = sr
        self.audio_length = audio_length
        self.mono = mono
        self.mono_type = mono_type
        self.start_time = start_time
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        filename = self.datalist[idx]
        # load and clip to duration
        waveform, sr = load_waveform(filename, target_sr=self.sr, start_time=self.start_time, duration=self.audio_length, mono=self.mono, mono_type=self.mono_type)
        # return stereo Tensor and sample key
        return waveform.float(), filename.stem
    
    def __len__(self):
        return len(self.datalist)

class ImageBindAudioDataset(Dataset):

    def __init__(self, datalist: List[Path], start_time: float = 0.0, audio_length: float = 8.0, mono_type: str = "mean"):
        self.datalist = datalist
        self.audio_length = audio_length
        self.mono_type = mono_type
        self.start_time = start_time

    # from ImageBind
    def load_and_transform_audio_data(
        self,
        audio_path,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,
        mean=-4.268,
        std=9.138,
    ):

        audio_outputs = []
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration,
                                                    clips_per_video=clips_per_video)

        # waveform, sr = torchaudio.load(audio_path)
        # if sample_rate != sr:
        #     waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        waveform, sr = load_waveform(audio_path, target_sr=sample_rate, start_time=self.start_time, duration=self.audio_length, mono=True, mono_type=self.mono_type)
        
        all_clips_timepoints = get_clip_timepoints(clip_sampler, waveform.size(1) / sr)
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate):int(clip_timepoints[1] * sample_rate),
            ]
            waveform_melspec = waveform2melspec(waveform_clip, sample_rate, num_mel_bins,
                                                target_length)
            all_clips.append(waveform_melspec)

        normalize = v2.Normalize(mean=[mean], std=[std])
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

        return torch.stack(audio_outputs, dim=0)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx: int):
        filename = self.datalist[idx]
        return self.load_and_transform_audio_data(filename), filename.stem


class SynchformerAudioDataset(Dataset):

    def __init__(self, datalist: List[Path], duration: float = 8.0, start_time: float = 0.0, mono_type: str = "mean"):
        self.datalist = datalist
        self.expected_length = int(16000 * duration)
        self.duration = duration
        self.resampler = {}
        self.mono_type = mono_type
        self.start_time = start_time

    def __len__(self):
        return len(self.datalist)

    def sample(self, idx: int):
        filename = self.datalist[idx]
        # waveform, sr = torchaudio.load(filename)
        waveform, sr = load_waveform(filename, target_sr=16000, start_time=self.start_time, duration=self.duration, mono=True, mono_type=self.mono_type)

        waveform = waveform.squeeze()
        # print(f'Loading {filename} with shape {waveform.shape} and sample rate {sr} expected length {self.expected_length}')

        return waveform, filename.stem

    def __getitem__(self, idx: int):
        while True:
            try:
                return self.sample(idx)
            except Exception as e:
                log.error(f'Error loading {self.datalist[idx]}: {e}')
                idx = (idx + 1) % len(self.datalist)
