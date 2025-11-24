
# FSAD computation


## Prerequisites

Download a pretrained model from [this URL](https://www.dropbox.com/scl/fi/7h9cwfhn12wo6n9euotvh/FreeMusic-StereoCRW-1024.pth.tar?rlkey=k7x0x7uydql611kfh1s4tcjqb&e=1&dl=0) and put it under `checkpoints/pretrained-models`


## Usage

The two folders contain the same number of audio files with identical names.

Run the automation script with two folder paths as arguments:

```bash
bash run_eval.sh
```

The script will:
1. Generate CSV files for both folders
2. Run visualization for each folder
3. Calculate ITD distances between the folders

## Example

```bash
bash run_eval.sh

Output:

Results for CRW mode:
KL-divergence 1.1118132758501538
[info] Overall ITD MSE: 37.614105127440794

Results for GCC mode:
KL-divergence 1.197748563271088
[info] Overall ITD MSE: 39.38635211678856

[info] fsad_score: 0.22983144932751726

```

## Important Notes

- Input folders should contain audio files
- The script will create temporary CSV files (`temp_1.csv` and `temp_2.csv`)
- Pkl files will be saved in the `temp_1` and `temp_2` directory


## Output

The script will generate:
- Distance calculations between the two folders
- PKL files containing ITD analysis data

## License

MIT

# Reference

If you find this repo useful, please cite our papers:

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

Please also cite the previous paper if you use the code in this repo. Thanks again for their great work.

```bibtex
@article{sun2024both,
  title={Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation},
  author={Sun, Peiwen and Cheng, Sitong and Li, Xiangtai and Ye, Zhen and Liu, Huadai and Zhang, Honggang and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2410.10676},
  year={2024}
}
```

```bibtex
@inproceedings{chen2022sound,
  title={Sound localization by self-supervised time delay estimation},
  author={Chen, Ziyang and Fouhey, David F and Owens, Andrew},
  booktitle={European Conference on Computer Vision},
  pages={489--508},
  year={2022},
  organization={Springer}
}
```

