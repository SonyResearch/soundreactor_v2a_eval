import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from av_bench.metrics import compute_fd, compute_isc, compute_kl, compute_mmd_optimized
from av_bench.synchformer.synchformer import Synchformer, make_class_grid

log = logging.getLogger()
device = 'cuda'
def clean_sample_name(sample_name: str) -> str:
    """
    Normalize a predicted sample name so it matches the GT key format.
    - Remove a trailing '_generated'
    - Remove a trailing '-123' or '_123'
    """
    # 1) strip "_generated"
    if sample_name.endswith("_generated"):
        sample_name = sample_name[: -len("_generated")]
    
    elif sample_name.endswith("_sfx"):
        sample_name = sample_name[: -len("_sfx")]

    # # 2) strip trailing "-<digits>" or "_<digits>"
    # sample_name = re.sub(r"[_-]\d+$", "", sample_name)

    return sample_name

def unroll_paired_dict(
    gt_dict: dict,
    pred_dict: dict,
    cat: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list]:
    gt_map   = {clean_sample_name(k): k for k in gt_dict.keys()}
    pred_map = {clean_sample_name(k): k for k in pred_dict.keys()}

    unpaired_ids = set(gt_map.keys()) ^ set(pred_map.keys())


    gt_tensors, pred_tensors = [], []
    for vid, pred_key in pred_map.items():
        if vid in unpaired_ids:          
            print(f"Sample {vid} not found in ground truth.")
            continue

        gt_key = gt_map[vid]         
        gt_tensors.append(gt_dict[gt_key])
        pred_tensors.append(pred_dict[pred_key])
    if cat:
        return torch.cat(gt_tensors, dim=0), torch.cat(pred_tensors,
                                                        dim=0), list(unpaired_ids)
    else:
        return torch.stack(gt_tensors, dim=0), torch.stack(pred_tensors,
                                                            dim=0), list(unpaired_ids)



def unroll_paired_dict_with_key(
    gt_d: dict,
    d: dict,
    key: str = 'logits',
    *,
    num_samples: Optional[int] = 10
) -> tuple[list[torch.Tensor], torch.Tensor]:

    gt_features = {}
    paired_features = defaultdict(list)

    for sample_name, features in gt_d.items():
        gt_features[sample_name] = features[key]

    for sample_name, features in d.items():
        sample_name = sample_name
        paired_features[sample_name].append(features[key])

    # find the number of samples
    for sample_name, features in paired_features.items():
        if num_samples is None:
            num_samples = len(features)
        else:
            assert num_samples <= len(features)

    # combine the two dictionaries
    gt_feat_list = []
    paired_feat_list = [[] for _ in range(num_samples)]
    for sample_name, features in paired_features.items():
        if sample_name not in gt_features:
            print(f'Sample {sample_name} not found in ground truth.')
            continue
        gt_feat_list.append(gt_features[sample_name])
        for i in range(num_samples):
            paired_feat_list[i].append(features[i])

    gt_feat_list = torch.stack(gt_feat_list, dim=0)
    paired_feat_list = [torch.stack(feat_list, dim=0) for feat_list in paired_feat_list]

    return paired_feat_list, gt_feat_list


def unroll_dict_all_keys(d: dict) -> dict[str, torch.Tensor]:
    out_dict = defaultdict(list)
    for k, v in d.items():
        for k2, v2 in v.items():
            out_dict[k2].append(v2)

    for k, v in out_dict.items():
        out_dict[k] = torch.stack(v, dim=0)

    return out_dict


def unroll_dict(d: dict, cat: bool = False) -> torch.Tensor:
    out_list = []
    for k, v in d.items():
        out_list.append(v)

    if cat:
        return torch.cat(out_list, dim=0)
    else:
        return torch.stack(out_list, dim=0)


@torch.inference_mode()
def evaluate(
    gt_audio_cache: Path,
    pred_audio_cache: Path,
    *,
    is_paired: bool = True,
    num_samples: int = 1,
    skip_video_related: bool = False,
    syncformer_ckpt_path: Path = '',
    mono_type: str = 'mean',
) -> Dict[str, float]:
    if not skip_video_related:
        sync_model = Synchformer().to(device).eval()
        sd = torch.load(syncformer_ckpt_path, weights_only=True)
        sync_model.load_state_dict(sd)

    gt_audio_cache = gt_audio_cache.expanduser()
    pred_audio_cache = pred_audio_cache.expanduser()

    gt_pann_features = torch.load(gt_audio_cache / f'pann_features_mono_{mono_type}.pth', weights_only=True)
    pred_pann_features = torch.load(pred_audio_cache / f'pann_features_mono_{mono_type}.pth', weights_only=True)

    gt_vggish_features = torch.load(gt_audio_cache / f'vggish_features_mono_{mono_type}.pth', weights_only=True)
    pred_vggish_features = torch.load(pred_audio_cache / f'vggish_features_mono_{mono_type}.pth', weights_only=True)

    gt_passt_features_pre = torch.load(gt_audio_cache / f'passt_features_embed_mono_{mono_type}.pth', weights_only=True)
    pred_passt_features = torch.load(pred_audio_cache / f'passt_features_embed_mono_{mono_type}.pth', weights_only=True)
    gt_passt_logits_pre = torch.load(gt_audio_cache / f'passt_logits_mono_{mono_type}.pth', weights_only=True)
    pred_passt_logits = torch.load(pred_audio_cache / f'passt_logits_mono_{mono_type}.pth', weights_only=True)

    
    gt_openl3_features = torch.load(gt_audio_cache / f'openl3_mono_{mono_type}.pth', weights_only=True)
    pred_openl3_features = torch.load(pred_audio_cache / f'openl3_mono_{mono_type}.pth', weights_only=True)
        
    gt_clap_features = torch.load(gt_audio_cache / f'clap_laion_audio_mono_{mono_type}.pth', weights_only=True)
    pred_clap_features = torch.load(pred_audio_cache / f'clap_laion_audio_mono_{mono_type}.pth', weights_only=True)
    # convert these dictionaries (with filenames as keys) to lists
    if is_paired:
        paired_panns_logits, gt_panns_logits = unroll_paired_dict_with_key(gt_pann_features,
                                                                           pred_pann_features,
                                                                           num_samples=num_samples)
        if not skip_video_related and (gt_audio_cache / 'imagebind_video.pth').exists():
            ib_video_features = torch.load(gt_audio_cache / 'imagebind_video.pth',
                                           weights_only=True)
            ib_audio_features = torch.load(pred_audio_cache / f'imagebind_audio_{mono_type}.pth',
                                           weights_only=True)
            paired_ib_video_features, paired_ib_audio_features, unpaired_ib_keys = unroll_paired_dict(
                ib_video_features, ib_audio_features)
            log.info(f'Unpaired IB features keys: {unpaired_ib_keys}')
        else:
            paired_ib_video_features = paired_ib_audio_features = None
            log.info('No IB features found, skipping IB-score evaluation')

        if not skip_video_related and (gt_audio_cache / 'synchformer_video.pth').exists():
            sync_video_features = torch.load(gt_audio_cache / 'synchformer_video.pth',
                                             weights_only=True)
            sync_audio_features = torch.load(pred_audio_cache / f'synchformer_audio_{mono_type}.pth',
                                             weights_only=True)
            paired_sync_video_features, paired_sync_audio_features, unpaired_sync_keys = unroll_paired_dict(
                sync_video_features, sync_audio_features)
            log.info(f'Unpaired Synchformer features keys: {unpaired_sync_keys}')
        else:
            paired_sync_video_features = paired_sync_audio_features = None
            log.info('No Synchformer features found, skipping DeSync evaluation')

        if (gt_audio_cache / 'clap_laion_text.pth').exists():
            laion_clap_text_features = torch.load(gt_audio_cache / 'clap_laion_text.pth',
                                                  weights_only=True)
            laion_clap_audio_features = torch.load(pred_audio_cache / f'clap_laion_audio_mono_{mono_type}.pth',
                                                   weights_only=True)
            paired_laion_clap_text_features, paired_laion_clap_audio_features, unpaired_laion_clap_keys = unroll_paired_dict(
                laion_clap_text_features, laion_clap_audio_features)
            log.info(f'Unpaired LAION CLAP features keys: {unpaired_laion_clap_keys}')
        else:
            paired_laion_clap_text_features = paired_laion_clap_audio_features = None
            log.info('No CLAP features found, skipping CLAP-score evaluation')
    else:
        paired_panns_logits = gt_panns_logits = None
        paired_ib_video_features = paired_ib_audio_features = None

    gt_vggish_features = unroll_dict(gt_vggish_features, cat=True)
    pred_vggish_features = unroll_dict(pred_vggish_features, cat=True)
    
    gt_openl3_features = unroll_dict(gt_openl3_features)
    gt_openl3_features = gt_openl3_features.reshape(-1, gt_openl3_features.shape[-1]).cpu()
    pred_openl3_features = unroll_dict(pred_openl3_features)
    pred_openl3_features = pred_openl3_features.reshape(-1, pred_openl3_features.shape[-1]).cpu()
    
    gt_clap_features = unroll_dict(gt_clap_features).cpu()
    pred_clap_features = unroll_dict(pred_clap_features).cpu()
    
    gt_pann_features = unroll_dict_all_keys(gt_pann_features)
    pred_pann_features = unroll_dict_all_keys(pred_pann_features)
    
    if is_paired:
        gt_passt_features, pred_passt_features, unpaired_passt_keys = unroll_paired_dict(
            gt_passt_features_pre, pred_passt_features)
        log.info(f'Unpaired PASST features keys: {unpaired_passt_keys}')
        gt_passt_logits, pred_passt_logits, unpaired_passt_keys = unroll_paired_dict(
            gt_passt_logits_pre, pred_passt_logits)
        log.info(f'Unpaired PASST logits keys: {unpaired_passt_keys}')
    else:
        gt_pann_features = unroll_dict(gt_pann_features).cpu()
        pred_pann_features = unroll_dict(pred_pann_features).cpu()
        gt_passt_features = unroll_dict(gt_passt_features_pre).cpu()
        pred_passt_features = unroll_dict(pred_passt_features).cpu()
    


    output_metrics = {}

    fd_vgg = compute_fd(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['FD-VGG'] = fd_vgg

    fd_pann = compute_fd(pred_pann_features['2048'].numpy(), gt_pann_features['2048'].numpy())
    output_metrics['FD-PANN'] = fd_pann

    fd_passt = compute_fd(pred_passt_features.numpy(), gt_passt_features.numpy())
    output_metrics['FD-PASST'] = fd_passt
    
    fd_openl3 = compute_fd(pred_openl3_features.numpy(), gt_openl3_features.numpy())
    output_metrics['FD-OpenL3'] = fd_openl3

    fd_lclap = compute_fd(pred_clap_features.numpy(), gt_clap_features.numpy())
    output_metrics['FD-LAION-CLAP'] = fd_lclap
    
    ### MMD ####
    mmd_vgg_2 = compute_mmd_optimized(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['MMD-VGG'] = mmd_vgg_2
    
    mmd_pann_2 = compute_mmd_optimized(pred_pann_features['2048'].numpy(), gt_pann_features['2048'].numpy())
    output_metrics['MMD-PANN'] = mmd_pann_2
    
    mmd_passt_2 = compute_mmd_optimized(pred_passt_features.numpy(), gt_passt_features.numpy())
    output_metrics['MMD-PASST'] = mmd_passt_2
    
    mmd_openl3_2 = compute_mmd_optimized(pred_openl3_features.numpy(), gt_openl3_features.numpy())
    output_metrics['MMD-OpenL3'] = mmd_openl3_2
    
    mmd_lclap_2 = compute_mmd_optimized(pred_clap_features.numpy(), gt_clap_features.numpy())
    output_metrics['MMD-LAION-CLAP'] = mmd_lclap_2
    
    #### MAUVE ####
    # mauve_score_openl3 = mauve.compute_mauve(p_features=pred_openl3_features, q_features=gt_openl3_features).mauve
    # # Take the negative log of the MAUVE score
    # ln_mauve_score_openl3 = - np.log(mauve_score_openl3)
    # output_metrics['MAUVE-OpenL3'] = ln_mauve_score_openl3
    
    # mauve_score_lclap = mauve.compute_mauve(p_features=pred_clap_features, q_features=gt_clap_features).mauve
    # # Take the negative log of the MAUVE score
    # ln_mauve_score_lclap = - np.log(mauve_score_lclap)
    # output_metrics['MAUVE-LAION-CLAP'] = ln_mauve_score_lclap
    
    # mauve_score_passt = mauve.compute_mauve(p_features=pred_passt_features, q_features=gt_passt_features).mauve
    # # Take the negative log of the MAUVE score
    # ln_mauve_score_passt = - np.log(mauve_score_passt)
    # output_metrics['MAUVE-PASST'] = ln_mauve_score_passt
    

    if is_paired:
        kl_metrics = compute_kl(paired_panns_logits, gt_panns_logits)
        output_metrics['KL-PANNS-softmax'] = kl_metrics['kl_softmax']

        kl_metrics = compute_kl([pred_passt_logits], gt_passt_logits)
        output_metrics['KL-PASST-softmax'] = kl_metrics['kl_softmax']

        metric_isc = compute_isc(
            pred_pann_features,
            feat_layer_name='logits',
            splits=10,
            samples_shuffle=True,
            rng_seed=2020,
        )
        output_metrics['ISC-PANNS-mean'] = metric_isc['inception_score_mean']
        output_metrics['ISC-PANNS-std'] = metric_isc['inception_score_std']

        metrics_isc = compute_isc(
            pred_passt_logits,
            feat_layer_name=None,
            splits=10,
            samples_shuffle=True,
            rng_seed=2020,
        )
        output_metrics['ISC-PASST-mean'] = metrics_isc['inception_score_mean']
        output_metrics['ISC-PASST-std'] = metrics_isc['inception_score_std']

    if is_paired and paired_ib_video_features is not None:
        # compute ib score
        ib_score = torch.cosine_similarity(paired_ib_video_features,
                                           paired_ib_audio_features,
                                           dim=-1).mean()
        output_metrics['IB-Score'] = ib_score.item()

    if is_paired and paired_sync_video_features is not None:
        # compute sync score
        batch_size = 16
        total_samples = paired_sync_video_features.shape[0]
        total_sync_scores = []
        sync_grid = make_class_grid(-2, 2, 21)
        for i in tqdm(range(0, total_samples, batch_size)):
            sync_video_batch = paired_sync_video_features[i:i + batch_size].to(device)
            sync_audio_batch = paired_sync_audio_features[i:i + batch_size].to(device)
            logits = sync_model.compare_v_a(sync_video_batch[:, :14], sync_audio_batch[:, :14])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

            logits = sync_model.compare_v_a(sync_video_batch[:, -14:], sync_audio_batch[:, -14:])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

        average_sync_score = np.mean(total_sync_scores)
        output_metrics['DeSync'] = average_sync_score

    if is_paired and paired_laion_clap_text_features is not None:
        # compute clap score
        clap_score = torch.cosine_similarity(paired_laion_clap_text_features.cpu(),
                                             paired_laion_clap_audio_features.cpu(),
                                             dim=-1).mean()
        output_metrics['LAION-CLAP-Score'] = clap_score.item()

    return output_metrics
