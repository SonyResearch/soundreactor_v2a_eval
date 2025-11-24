from pathlib import Path

import laion_clap
import torch
import torch.nn as nn
import torchaudio
from hear21passt.base import get_basic_model, get_model_passt
from imagebind.models import imagebind_model
# from msclap import CLAP

from av_bench.panns import Cnn14
from av_bench.synchformer.synchformer import Synchformer
from av_bench.vggish.vggish import VGGish
import torchopenl3

class ExtractionModels(nn.Module):

    def __init__(
        self,
        clap_model_path: Path,
        syncformer_ckpt_path: Path,
    ):
        super().__init__()

        features_list = ["2048", "logits"]
        self.panns = Cnn14(
            features_list=features_list,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        self.panns = self.panns.eval()
        self.vggish = VGGish(postprocess=False).eval()

        # # before the prediction head
        # # https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/models/passt.py#L440-L441
        self.passt_model = get_basic_model(mode="all")
        self.passt_model.eval()
        self.openl3 = torchopenl3

        self.imagebind = imagebind_model.imagebind_huge(pretrained=True).eval()
        
        path_str = str(clap_model_path)
        enable_fusion = "fusion" in path_str
        amodel = "HTSAT-base" if "music_speech_audioset_epoch_15_esc_89" in path_str else "HTSAT-tiny"
        self.laion_clap = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel).cuda().eval()
        self.laion_clap.load_ckpt(clap_model_path, verbose=False)
        self.laion_clap.eval()

        self.synchformer = Synchformer().eval()
        sd = torch.load(syncformer_ckpt_path, weights_only=True)
        self.synchformer.load_state_dict(sd)

        # from synchformer
        self.sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=128,
        )
