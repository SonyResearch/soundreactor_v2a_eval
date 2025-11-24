import logging
from argparse import ArgumentParser
from pathlib import Path

import laion_clap
import pandas as pd
import torch
from colorlog import ColoredFormatter
from tqdm import tqdm

log = logging.getLogger()
device = 'cuda'

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


setup_eval_logging()


@torch.inference_mode()
def extract(args):
    text_csv = args.text_csv
    output_cache_path = args.output_cache_path
    clap_ckpt_path = args.clap_ckpt_path

    output_cache_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(text_csv).to_dict(orient='records')

    path_str = str(clap_ckpt_path)
    enable_fusion = "fusion" in path_str
    amodel = "HTSAT-base" if "music_speech_audioset_epoch_15_esc_89" in path_str else "HTSAT-tiny"
    laion_clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel).cuda().eval()
    laion_clap_model.load_ckpt(clap_ckpt_path, verbose=False)

    all_laion_clap = {}
    for row in tqdm(df):
        name = str(row['name'])
        caption = row['caption']

        text_data = [caption]
        text_embed = laion_clap_model.get_text_embedding(text_data, use_tensor=True)
        all_laion_clap[name] = text_embed.cpu().squeeze()

    torch.save(all_laion_clap, output_cache_path / 'clap_laion_text.pth')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--text_csv', type=Path, required=True)
    parser.add_argument('--output_cache_path', type=Path, required=True)
    parser.add_argument('--clap_ckpt_path', type=Path, required=True)
    args = parser.parse_args()
    extract(args)
