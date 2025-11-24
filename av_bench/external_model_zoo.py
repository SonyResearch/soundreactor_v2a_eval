import importlib.util
import logging
import math
import os
from abc import ABC, abstractmethod
from email.policy import strict
from pathlib import Path
from turtle import st
from typing import Literal, Optional, Union

import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
from hypy_utils.downloader import download_file
from packaging import version
from torch import nn

log = logging.getLogger(__name__)


class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int, audio_len: Optional[Union[float, int]] = None):
        self.audio_len = audio_len
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @torch.no_grad()
    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        if not embd.shape[-1] == self.num_features:
            raise RuntimeError(f"[{self.name}]: Expected {self.num_features} features, got {embd.shape[-1]}")
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd
    
    def postprocess_resoultion(self, audio: np.ndarray, emb: np.ndarray, pooling_resolution_sec: int = 1) -> np.ndarray:
        audio_dur = audio.shape[0] / self.sr
        pooling_resoultion = audio_dur / pooling_resolution_sec
        stride = int(emb.shape[0] / pooling_resoultion)
        emb = emb.unfold(0, stride, stride).mean(-1)
        return emb

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass


class MERTModel(ModelLoader):
    """
    MERT model from https://huggingface.co/m-a-p/MERT-v1-330M

    Please specify the layer to use (1-12).
    """
    def __init__(self, size='v1-95M', layer=12, limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        super().__init__(f"MERT-{size}" + ("" if layer == 12 else f"-{layer}"), 768, 24000, audio_len=audio_len)
        self.huggingface_id = f"m-a-p/MERT-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr
        
    def load_model(self):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        
        self.model = AutoModel.from_pretrained(self.huggingface_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.huggingface_id, trust_remote_code=True)
        # self.sr = self.processor.sampling_rate
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to 9 minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze() # [13 layers, timeframes, 768]
            out = out[self.layer] # [timeframes, 768]
            out = self.postprocess_resoultion(audio, out)
        return out

class W2V2Model(ModelLoader):
    """
    W2V2 model from https://huggingface.co/facebook/wav2vec2-base-960h, https://huggingface.co/facebook/wav2vec2-large-960h

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"w2v2-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"facebook/wav2vec2-{size}-960h"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, Wav2Vec2Model
        
        self.model = Wav2Vec2Model.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class HuBERTModel(ModelLoader):
    """
    HuBERT model from https://huggingface.co/facebook/hubert-base-ls960, https://huggingface.co/facebook/hubert-large-ls960

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"hubert-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"facebook/hubert-{size}-ls960"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, HubertModel

        self.model = HubertModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class WavLMModel(ModelLoader):
    """
    WavLM model from https://huggingface.co/microsoft/wavlm-base, https://huggingface.co/microsoft/wavlm-base-plus, https://huggingface.co/microsoft/wavlm-large

    Please specify the model size ('base', 'base-plus', or 'large') and the layer to use (1-12 for 'base' or 'base-plus' and 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'base-plus', 'large'], layer: Literal['12', '24'], limit_minutes=6, audio_len: Optional[Union[float, int]] = None):
        model_dim = 768 if size in ['base', 'base-plus'] else 1024
        model_identifier = f"wavlm-{size}" + ("" if (layer == 12 and size in ['base', 'base-plus']) or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"patrickvonplaten/wavlm-libri-clean-100h-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, WavLMModel

        self.model = WavLMModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]
            out = self.postprocess_resoultion(audio, out)

        return out


class WhisperModel(ModelLoader):
    """
    Whisper model from https://huggingface.co/openai/whisper-base
    
    Please specify the model size ('tiny', 'base', 'small', 'medium', or 'large').
    """
    def __init__(self, size: Literal['tiny', 'base', 'small', 'medium', 'large'], audio_len: Optional[Union[float, int]] = None):
        dimensions = {
            'tiny': 384,
            'base': 512,
            'small': 768,
            'medium': 1024,
            'large': 1280
        }
        model_dim = dimensions.get(size)
        model_identifier = f"whisper-{size}"

        super().__init__(model_identifier, model_dim, 16000, audio_len=audio_len)
        self.huggingface_id = f"openai/whisper-{size}"
        
    def load_model(self):
        from transformers import AutoFeatureExtractor, WhisperModel
        
        self.model = WhisperModel.from_pretrained(self.huggingface_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        input_features = inputs.input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state # [1, timeframes, 512]
            out = out.squeeze() # [timeframes, 384 or 512 or 768 or 1024 or 1280]

        return out