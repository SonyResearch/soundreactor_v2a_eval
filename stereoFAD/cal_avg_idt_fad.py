import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from scipy import linalg
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from torch.utils.data import DataLoader
# from fad import FrechetAudioDistance
import torch.nn.functional as F
import torchaudio
sys.path.append("v2a_eval/stereoFAD/")


def pad_short_audio(audio, min_samples=32000):
    if(audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.size(-1)), mode='constant', value=0.0)
    return audio


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datadir,
        sr=16000,
        limit_num=None,
    ):
        self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
        self.datalist = sorted(self.datalist)
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self.sr = sr

    def __getitem__(self, index):
        while True:
            try:
                filename = self.datalist[index]
                waveform = self.read_from_file(filename)
                if waveform.size(-1) < 1:
                    raise ValueError("empty file %s" % filename)
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.datalist)
        
        return waveform, os.path.basename(filename)

    def __len__(self):
        return len(self.datalist)

    def read_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # Only use the first channel
        audio = audio[0:1,...]
        audio = audio - audio.mean()

        # if file_sr != self.sr and file_sr == 32000 and self.sr == 16000:
        #     audio = audio[..., ::2]
        # if file_sr != self.sr and file_sr == 48000 and self.sr == 16000:
        #     audio = audio[..., ::3]
        # el

        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=file_sr, new_freq=self.sr, # rolloff=0.95, lowpass_filter_width=16 
            )
            # audio = torch.FloatTensor(librosa.resample(audio.numpy(), file_sr, self.sr))
            
        audio = pad_short_audio(audio, min_samples=32000)
        return audio

class FrechetAudioDistance:
    def __init__(
        self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8
    ):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def load_audio_data(self, x):
        outputloader = DataLoader(
            WaveDataset(
                x,
                16000,
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=8,
        )
        data_list = []
        print("Loading data to RAM")
        for batch in tqdm(outputloader):
            data_list.append((batch[0][0,0], 16000))
        return data_list

    def get_embeddings(self, x, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        x = self.load_audio_data(x)
        if isinstance(x, list): 
            try:
                for audio, sr in tqdm(x, disable=(not self.verbose)):
                    embd = self.model.forward(audio.numpy(), sr)
                    if self.model.device == torch.device("cuda"):
                        embd = embd.cpu()
                    embd = embd.detach().numpy()
                    embd_lst.append(embd)
            except Exception as e:
                print(
                    "[Frechet Audio Distance] get_embeddings throw an exception: {}".format(
                        str(e)
                    )
                )
        else:
            raise AttributeError

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def score(self, background_dir, eval_dir, store_embds=False, limit_num=None, recalculate = False): 
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        try:
            fad_target_folder_cache = eval_dir + "_fad_feature_cache.npy"
            fad_generated_folder_cache = background_dir + "_fad_feature_cache.npy"

            if(not os.path.exists(fad_generated_folder_cache) or recalculate):
                embds_background = self.get_embeddings(background_dir, limit_num=limit_num)
                np.save(fad_generated_folder_cache, embds_background)
            else:
                print("Reload fad_generated_folder_cache", fad_generated_folder_cache)
                embds_background = np.load(fad_generated_folder_cache)

            if(not os.path.exists(fad_target_folder_cache) or recalculate):
                embds_eval = self.get_embeddings(eval_dir, limit_num=limit_num)
                np.save(fad_target_folder_cache, embds_eval)
            else:
                print("Reload fad_target_folder_cache", fad_target_folder_cache)
                embds_eval = np.load(fad_target_folder_cache)

            if store_embds:
                np.save("embds_background.npy", embds_background)
                np.save("embds_eval.npy", embds_eval)

            if len(embds_background) == 0:
                print(
                    "[Frechet Audio Distance] background set dir is empty, exitting..."
                )
                return -1

            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exitting...")
                return -1

            mu_background, sigma_background = self.calculate_embd_statistics(
                embds_background
            )
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, sigma_background, mu_eval, sigma_eval
            )

            return {"frechet_audio_distance": fad_score}

        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            return -1

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_mse(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

folder1 = 'temp_1'
folder2 = 'temp_2'

mse_results = {}
def softmax_normalize(x):
    return F.softmax(torch.tensor(x), dim=-1)
data_list1, data_list2 = [], []
for filename in os.listdir(folder2):
    if filename.endswith('.pkl'):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        # print(file1_path,file2_path)
        if os.path.exists(file1_path):
            # Apply softmax normalization to feat_itds
            data1 = load_pickle(file1_path)
            # data1["feat_itds"]=softmax_normalize(data1["feat_itds"])
            data2 = load_pickle(file2_path)
            # data2["feat_itds"]=softmax_normalize(data2["feat_itds"])
            data_list1.append(data1["feat_itds"])
            data_list2.append(data2["feat_itds"])

data_list1 = torch.stack(data_list1, dim=0)
data_list2 = torch.stack(data_list2, dim=0)
# data_list2 = torch.flip(data_list2, dims=[1])

data_list1 = torch.mean(data_list1.view(data_list1.shape[0], 20, 5, 256), dim=2).cpu().numpy()
data_list1 = data_list1.reshape(data_list1.shape[0], -1)
data_list2 = torch.mean(data_list2.view(data_list2.shape[0], 20, 5, 256), dim=2).cpu().numpy()
data_list2 = data_list2.reshape(data_list2.shape[0], -1)

fad = FrechetAudioDistance()

mu_background, sigma_background = fad.calculate_embd_statistics(data_list1)
mu_eval, sigma_eval = fad.calculate_embd_statistics(data_list2)

fad_score = fad.calculate_frechet_distance(mu_background, sigma_background, mu_eval, sigma_eval)

print("[info] fsad_score:", fad_score * 1000)
