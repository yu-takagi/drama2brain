# Unveiling Multi-level and Multi-modal Semantic Representations in the Human Brain using Large Language Models
Yuko Nakagi ¹, Takuya Matsuyama ¹, Naoko Koide-Majima, Hiroto Q. Yamaguchi, Rieko Kubo, Shinji Nishimoto ² and Yu Takagi ², EMNLP 2024
(¹ Equal first auther, ² Equal last auther)

[[Paper](https://aclanthology.org/2024.emnlp-main.1133/)]
[[Project Page](https://sites.google.com/view/llm-and-brain/%E3%83%9B%E3%83%BC%E3%83%A0)]
[[Dataset](https://openneuro.org/datasets/ds005531/versions/1.0.0)]

This is a repository to reproduce the methods of our study on Multi-level and Multi-modal Semantic Representations in the Human Brain using Large Language Models that we published.

LLMs are now being used to study semantic representations in the human brain. However, most previous research has focused on single aspects such as speech or object content, overlooking the multi-level nature of semantic processing in real life.
<p align="center">
<img src=/images/overview.png />
</p>

To address this, we collected densely annotated fMRI datasets from participants watching over 8 hours of video. We then extracted latent representations from LLMs and multimodal LLMs. We built brain encoding models to predict brain activity from the latent representations.
<p align="center">
<img src=/images/methods.png />
</p>

We confirmed that the latent representations of different semantic content correspond to spatially distinct brain regions.
<p align="center">
<img src=/images/result1.png />
</p>

We also confirmed that Multimodal features correspond to brain activity better and more uniquely than the unimodal models.
<p align="center">
<img src=/images/result2.png />
</p>

Ensure that the necessary data is placed in `./data` and set the path for `drama2brain/data_const.py` according to your environment.
## Performing Ridge Regression with a Single Feature
```sh
python -m drama2brain.run_training_encoder --subject_name DM01\
                                            --time_delays 8 10\
                                            --modalities semantic\
                                            --modality_hparams speechTranscription_cw1_avgToken_ALL\
                                            --feat_names GPT2\
                                            --select_layers all\
                                            --device 0\
                                            --reg_type mono\
                                            --perm_type block\
                                            --reduce_dim default None\
                                            --dataset all
```

Below are explanations for some of the arguments. Full details are available in `drama2brain/run_training_encoder.py`:
- Delay time (`time_delays`): 3-5 seconds
- Modality-specific parameters (`modality_hparams`): speechTranscription_cw1_avgToken_ALL
    - Expressed by concatenating various parameters used for feature creation with underscores (`_`)
        - Annotation type: speechTranscription
        - Time window: cw1
        - Token handling: avgToken
        - Annotator used: ALL
- Model used for feature extraction (`feat_names`): GPT2

The following files will be primarily saved:
- `cv_cc_raw.npy`: Prediction accuracy from cross-validation
- `cc_raw.npy`: Prediction accuracy on test data
- `coef_raw.npy`: Model weights
- `pvalues_corrected_raw.npy`: p-values from statistical tests
- `dist_raw.png`: Histogram of prediction accuracy on test data

(Note: The `raw` part of the file names may change depending on the dimensionality reduction method used.)

## Performing Banded Ridge Regression with Multiple Features
```sh
python3 -m drama2brain.run_training_encoder --subject_name DM01\
                                            --time_delays 8 10\
                                            --modalities semantic semantic semantic semantic semantic\
                                            --modality_hparams speechTranscription_cw1_avgToken_ALL objectiveAnnot50chara_cw1_avgToken_ALL story_cw1_avgToken_ALL eventContent_cw1_avgToken_ALL timePlace_cw1_avgToken_ALL\
                                            --feat_names Llama2 Llama2 Llama2 Llama2 Llama2\
                                            --select_layers 32 13 17 20 2\
                                            --device 0\
                                            --reduce_dim pca 1280\
                                            --reg_type multi\
                                            --perm_type block\
                                            --n_iter 100\
                                            --dataset all
```
Full argument explanations are available in `drama2brain/run_training_encoder.py`.

> [!WARNING]
> - The arguments `modalities`, `modality_hparams`, `feat_names`, and `select_layers` correspond to each other in parallel. In the above example, the first feature uses the conditions `modalities`=semantic, `modality_hparams`=speechTranscription_cw1_avgToken_ALL, `feat_names`=Llama2, and `select_layers`=32.
> - Currently, only one `time_delays` and `reduce_dim` can be selected. The same delay time and dimensionality reduction are applied to all features.

The following files will be saved:
- `cc_1280PCs.npy`: Prediction accuracy on test data
- `coef_1280PCs.npy`: Model weights
- `pvalues_corrected_1280PCs.npy`: p-values from statistical tests
- `dist_1280PCs.png`: Histogram of prediction accuracy on test data
- `preds_1280PCs.npy`: Predicted brain activity on test data