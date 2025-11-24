# audioldm_eval

import numpy as np
from scipy import linalg
from typing import Union, Iterable
import torch

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    Adapted from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py
    
    Numpy implementation of the Frechet Distance.
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1: Embedding's mean statistics for generated samples.
    -- mu2: Embedding's mean statistics for reference samples.
    -- sigma1: Covariance matrix over embeddings for generated samples.
    -- sigma2: Covariance matrix over embeddings for reference samples.
    Returns:
    --  Fréchet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def ensure_tensor(x: np.ndarray | torch.Tensor, device=None):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(device, non_blocking=True) if device else x

# def _frechet_distance(
#     mu_x: torch.Tensor,
#     sigma_x: torch.Tensor,
#     mu_y: torch.Tensor,
#     sigma_y: torch.Tensor,
#     device=None,
# ) -> torch.Tensor:
#     # https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
#     mu_x = ensure_tensor(mu_x, device)
#     sigma_x = ensure_tensor(sigma_x, device)
#     mu_y = ensure_tensor(mu_y, device)
#     sigma_y = ensure_tensor(sigma_y, device)
#     a = (mu_x - mu_y).square().sum(dim=-1)
#     b = sigma_x.trace() + sigma_y.trace()
#     c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)
#     return a + b - 2 * c
def compute_fd(embds_background, embds_eval):
    # embds_background == generated
    # eval == ground_truth (from audioeval_ldm)

    mu_background, sigma_background = calculate_embd_statistics(embds_background)
    mu_eval, sigma_eval = calculate_embd_statistics(embds_eval)

    try:
        fad_score = calculate_frechet_distance(mu_eval, sigma_eval, mu_background, sigma_background)
        # fad_score = _frechet_distance(mu_eval, sigma_eval, mu_background, sigma_background)

    except ValueError as e:
        print(f"Error in FAD computation: {e}")
        fad_score = -1

    return fad_score
