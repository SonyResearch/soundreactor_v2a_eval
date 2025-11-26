# Evaluation toolkit for video-to-audio generation

Evaluation toolkit from [SoundReactor](https://arxiv.org/abs/2510.02110)

### Contanct: Koichi Saito: koichi.saito@sony.com

## Overview

This repository supports the evaluations of:

- Fréchet Distances (FD)
    - FD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
        - For 16kHz
    - FD_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
        - For 32kHz
    - FD_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
        - For 32kHz
    - FD_OpenL3, with [OpenL3](https://github.com/torchopenl3/torchopenl3)
        - For 48kHz
    - FD_L-CLAP, with [LAION-CLAP](https://github.com/LAION-AI/CLAP)
        - For 48kHz
    
    - You can refer [FADTK](https://github.com/microsoft/fadtk) or [KADTK](https://github.com/YoonjinXD/kadtk/) for choosing pretrained backbone of audio encoder.

- Maximum Mean Discrepancy (MMD) [Image](https://arxiv.org/pdf/2401.09603), [Music](https://arxiv.org/abs/2503.16669), [Audio](https://arxiv.org/abs/2502.15602)
    - MMD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
        - For 16kHz
    - MMD_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
        - For 32kHz
    - MMD_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
        - For 32kHz
    - MMD_OpenL3, with [OpenL3](https://github.com/torchopenl3/torchopenl3)
        - For 48kHz
    - MMD_L-CLAP, with [LAION-CLAP](https://github.com/LAION-AI/CLAP)
        - For 48kHz
        
    - You can refer [FADTK](https://github.com/microsoft/fadtk) or [KADTK](https://github.com/YoonjinXD/kadtk/) for choosing pretrained backbone of audio encoder.

- Fréchet Stereo Audio Distances [FSAD](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open)
    - FD with [StereoCRW](https://github.com/IFICL/stereocrw)
    - Please check [here](stereoFAD/README.md) to run FSAD.

- Inception Scores (IS)

    - IS_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
        - For 32kHz
    - IS_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
        - For 32kHz

- Mean KL Distances (MKL)

    - KL_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
        - For 32kHz
    - KL_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
        - For 32kHz

- CLAP Scores

    - LAION_CLAP, cosine similarity between text and audio embeddings computed by [LAION-CLAP](https://github.com/LAION-AI/CLAP).

- ImageBind Score
    
    Cosine similarity between video and audio embeddings computed by [ImageBind](https://github.com/facebookresearch/ImageBind), sometimes scaled by 100


- DeSync Score

    Average misalignment (in seconds) predicted by [Synchformer](https://github.com/v-iashin/Synchformer) with the `24-01-04T16-39-21` model trained on AudioSet. We average the results from the first 4.8 seconds and last 4.8 seconds of each video-audio pair.

## Installation

### 1. docker (recommended) 
Install docker and build docker container.
You can build dockerfile is located at `container/dockerfile`.

```sh
docker build -t tag .
```

### 2. miniforge
Or you can install via [miniforge](https://github.com/conda-forge/miniforge).
Yaml file is located at `container/environment.yml`.

```sh
mamba env create -f environment.yml
```

Then

```sh
mamba activate v2a_eval
```

Then install pytorch and flash-attn (we only tested on this version but it might work on different one.)
```sh
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.4.1 --no-build-isolation
```

### For Video Evaluation

If you plan to evaluate on videos, you will also need `ffmpeg`. Note that torchaudio imposes a maximum version limit (`ffmpeg<7`).

## Download Pretrained Models

- Download [LAION-CLAP checkpoint](https://github.com/LAION-AI/CLAP). Specify path to the checkpoint at `--clap_ckpt_path` in the shell script.
- Download [Synchformer](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth). Specify path to the checkpoint at `--syncformer_ckpt_path` in the shell script.
- Download [StereoCRW](https://www.dropbox.com/scl/fi/7h9cwfhn12wo6n9euotvh/FreeMusic-StereoCRW-1024.pth.tar?rlkey=k7x0x7uydql611kfh1s4tcjqb&e=1&dl=0). Put the checkpoint under `stereoFAD/checkpoints/pretrained-models`.


## Usage

### Overview

Evaluation is a two-stage process:

1. **Extraction**: extract video/text/audio features for ground-truth and audio features for the predicted audios.
2. **Evaluation**: compute the desired metrics using the extracted features.

*For FSAD, please check [here](stereoFAD/README.md).

### Extraction

#### 1. **Video feature extraction (optional).**
For video-to-audio applications, visual features are extracted from input videos.
**Input requirements:**

- Videos in .mp4 format (any FPS or resolution).
- Video names must match the corresponding audio file names (excluding extensions).

Run the following to extract visual features using `Synchformer` and `ImageBind`:

```sh
bash extract_video.sh
```

#### 2. **Text feature extraction (optional).**
For applications using text, text features are extracted from input text data.

**Input requirements:**

- A CSV file with at least two columns with a header row:
    - `name`: Matches the corresponding audio file name (excluding extensions).
    - `caption`: The text associated with the audio.

Run the following to extract text features using `LAION-CLAP`:

```sh
bash extract_text.sh
```

### Evaluation
You can run evaluation via

```sh
bash evaluate.sh
```

Since pretrained audio feature extractors are trained with mono audio except for [StereoCRW](https://github.com/IFICL/stereocrw), which we can compute FSAD separately at `stereoFAD/run_eval.sh`, audio samples are loaded as mono.

You can spesify mono type via `--mono_type` such as 'mean', 'left', 'right', and 'side' (difference between left and right).

If you don't want to recompute audio feature extractions, you can skip them by excluding
 - `--recompute_gt_cache` for ground-truth audio features.
 - `--recompute_pred_cache` for predicted audio features.

If there is no extracted features on text and videos, metrics related to those features are automatically skipped.


## Citation

To cite this repository, please use the following BibTeX entry:

```bibtex
@article{saito2025soundreactor,
  title={SoundReactor: Frame-level Online Video-to-Audio Generation},
  author={Koichi Saito and Julian Tanke and Christian Simon and Masato Ishii and Kazuki Shimada and Zachary Novack and Zhi Zhong and Akio Hayakawa and Takashi Shibuya and Yuki Mitsufuji},
  year={2025},
  eprint={2510.02110},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2510.02110}, 
  journal={arXiv preprint arXiv:2510.02110},
}
```

## Acknowledgment
https://github.com/hkchengrex/av-benchmark

https://github.com/PeiwenSun2000/Both-Ears-Wide-Open/

https://github.com/microsoft/fadtk/

https://github.com/YoonjinXD/kadtk/
