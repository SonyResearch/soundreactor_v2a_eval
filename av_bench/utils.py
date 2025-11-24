from collections import defaultdict
from typing import Literal, Optional

import torch
import re


# def clean_sample_name(sample_name: str) -> str:
#     # implement your own cleaning function here to map the sample name to the sample name
#     if len(sample_name) == len('000000014_zxpo56cpUBU_000007-0'):
#         # extract the zxpo56cpUBU portion
#         example = '000000014_zxpo56cpUBU_000007-0'
#         example_target = 'zxpo56cpUBU_000007'
#         start = example.find(example_target)
#         end = start + len(example_target)

#         vid_name = sample_name[start:end]
#     elif len(sample_name) == len('zxpo56cpUBU_000007-0'):
#         # extract the zxpo56cpUBU portion
#         example = 'zxpo56cpUBU_000007-0'
#         example_target = 'zxpo56cpUBU_000007'
#         start = example.find(example_target)
#         end = start + len(example_target)

#         vid_name = sample_name[start:end]

#     elif len(sample_name) == len('Y---g-f_I2yQ_000001_0'):
#         # extract the Y---g-f_I2yQ portion
#         example = 'Y---g-f_I2yQ_000001_0'
#         example_target = '---g-f_I2yQ_000001'
#         start = example.find(example_target)
#         end = start + len(example_target)

#         vid_name = sample_name[start:end]

#     elif len(sample_name) == len('zxpo56cpUBU_000007'):
#         vid_name = sample_name

#     else:
#         return sample_name

#     return vid_name

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

def unroll_paired_dict_with_key(gt_d: dict,
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
        breakpoint()
        if sample_name not in gt_features:
            print(f'Sample {sample_name} not found in ground truth.')
            continue
        gt_feat_list.append(gt_features[sample_name])
        for i in range(num_samples):
            paired_feat_list[i].append(features[i])

    gt_feat_list = torch.stack(gt_feat_list, dim=0)
    paired_feat_list = [torch.stack(feat_list, dim=0) for feat_list in paired_feat_list]

    return paired_feat_list, gt_feat_list

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

# def unroll_paired_dict(gt_dict: dict,
#                        pred_dict: dict,
#                        cat: bool = False) -> tuple[torch.Tensor, torch.Tensor, list]:
#     pred_keys_to_sample = {k: clean_sample_name(k) for k in pred_dict.keys()}
#     gt_keys_to_sample = {k: clean_sample_name(k) for k in gt_dict.keys()}
#     unpaired_samples = set(gt_keys_to_sample.keys()) ^ set(pred_keys_to_sample.keys())
    

#     gt_out_list = []
#     pred_out_list = []
#     for key, sample_name in pred_keys_to_sample.items():
#         if sample_name in unpaired_samples:
            
#             print(f'Sample {sample_name} not found in ground truth.')
#             continue
#         gt_key = gt_map[vid] 
#         gt_out_list.append(gt_dict[sample_name])
#         pred_out_list.append(pred_dict[key])

#     if cat:
#         return torch.cat(gt_out_list, dim=0), torch.cat(pred_out_list,
#                                                         dim=0), list(unpaired_samples)
#     else:
#         return torch.stack(gt_out_list, dim=0), torch.stack(pred_out_list,
#                                                             dim=0), list(unpaired_samples)


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
