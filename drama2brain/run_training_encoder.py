"""
Functions for training a model from llms' embeddings to brain activity.

- For Mono Ridge
python3 -m drama2brain.run_training_encoder --subject_name DM01 --time_delays 8 10 --modalities semantic --modality_hparams speechTranscription_cw1_avgToken_ALL --feat_names GPT2 --select_layers 1 --device 4 --reg_type mono --perm_type block --dataset all

- For Banded Ridge
python3 -m drama2brain.run_training_encoder --subject_name DM01 --time_delays 3 5 --modalities semantic semantic semantic semantic semantic --modality_hparams speechTranscription_cw1_avgToken_ALL objectiveAnnot50chara_cw1_avgToken_ALL story_cw1_avgToken eventContent_cw1_avgToken_ALL timePlace_cw1_avgToken_ALL --feat_names BERT BERT BERT BERT BERT --select_layers 12 6 9 10 7 --device 0 --reduce_dim default None --reg_type multi --perm_type block --n_iter 100
"""

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    ColumnKernelizer,
    Kernelizer,
    MultipleKernelRidgeCV,
)
from himalaya.ridge import BandedRidgeCV, ColumnTransformerNoStack
from himalaya.scoring import correlation_score, correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drama2brain.fmri_loader import collect_fmri_wholevoxels
from drama2brain.stim_loader import collect_stim, collect_stim_for_dimreducer

from drama2brain.utils import TrnVal, gen_nulldistrib_gauss, gen_nulldistrib_block, fdr_correction, make_rearanging_index_for_cv, make_cv, make_himalaya_pipeline
from drama2brain import data_const
    
def load_resp_wholevoxels(subject_name, dataset="all"):
    resp_trn, video_names_trn = collect_fmri_wholevoxels(subject_name,
                                                         trainvalid="TRAIN",
                                                         dataset=dataset)
    resp_val, video_names_val = collect_fmri_wholevoxels(subject_name,
                                                         trainvalid="VALID",
                                                         dataset=dataset)

    return TrnVal(trn=resp_trn, val=resp_val), TrnVal(trn=video_names_trn, val=video_names_val)


def load_stim(feat_path, reduce_dim, time_delays, dataset="all"):

    if reduce_dim[0] == "default":
        stim_trn = collect_stim(feat_path,
                            trainvalid="TRAIN",
                            ishrfexpand=True,
                            time_delays=time_delays,
                            dataset=dataset
                            )
        stim_val = collect_stim(feat_path,
                            trainvalid="VALID",
                            ishrfexpand=True,
                            time_delays=time_delays,
                            dataset=dataset
                            )
        return TrnVal(trn=stim_trn, val=stim_val)

    else:
        stim = collect_stim_for_dimreducer(
            feat_path,
            reduce_dim,
            ishrfexpand=True,
            time_delays=time_delays,
            dataset=dataset
            )

        return stim


def make_filename(reduce_dim, dataset="all"):
    if reduce_dim[0] == "pca":
        filename = f"{reduce_dim[1]}PCs"
    elif reduce_dim[0] == "srp":
        filename = f"{reduce_dim[1]}eps"
    else:
        filename = f"raw"

    if dataset != "all":
        filename = f"{filename}_{dataset}"
    
    return filename


def check_saved_score(scores_save_dir, reduce_dim, feat_path, dataset):
    if reduce_dim[0]=="pca":
        if dataset == "all":
            if os.path.exists(f"{scores_save_dir}/cc_{reduce_dim[1]}PCs.npy"):
                print(f"{feat_path} PCs' encoding results are already exist.")
                return True
            else:
                return False
        else:
            if os.path.exists(f"{scores_save_dir}/cc_{reduce_dim[1]}PCs_{dataset}.npy"):
                print(f"{feat_path} PCs' encoding results are already exist.")
                return True
            else:
                return False
        
    elif reduce_dim[0]=="srp":
        if dataset == "all":
            if os.path.exists(f"{scores_save_dir}/cc_{reduce_dim[1]}eps.npy"):
                print(f"{feat_path} SRP' encoding results are already exist.")
                return True
            else:
                return False
        else:
            if os.path.exists(f"{scores_save_dir}/cc_{reduce_dim[1]}eps_{dataset}.npy"):
                print(f"{feat_path} SRP' encoding results are already exist.")
                return True
            else:
                return False
        
    else:
        if dataset == "all":
            if os.path.exists(f"{scores_save_dir}/cc_raw.npy"):
                print(f"{feat_path} raw's encoding results are already exist.")
                return True
            else:
                return False
        else:
            if os.path.exists(f"{scores_save_dir}/cc_raw_{dataset}.npy"):
                print(f"{feat_path} raw's encoding results are already exist.")
                return True
            else:
                return False


def mono_regressor(
    stim: TrnVal[np.ndarray],
    resp: TrnVal[np.ndarray],
    emb_name: str,
    device: int,
    perm_type: str,
    cv_split_points: list[int],
    save_pred: bool,
) -> tuple[dict[str, np.ndarray], object]:
    """Train an encoder for mono feature space."""
    alphas = np.logspace(-12, 12, 25)
    if save_pred:
        cv, nested_run_onsets = make_cv(cv_split_points,
                                        return_nested_run_onsets=save_pred)
    else:
        cv = make_cv(cv_split_points)
    x_trn, x_val = stim.trn.astype("float32"), stim.val.astype("float32")
    y_trn, y_val = resp.trn.astype("float32"), resp.val.astype("float32")

    n_samples_val = x_val.shape[0]

    if device >= 0:
        if device < torch.cuda.device_count():
            torch.cuda.set_device(f"cuda:{device}")
            backend = set_backend("torch_cuda", on_error="warn")
            print("Running on GPU...")

        else:
            print("The CUDA device you specified is not available.")
            print("Running on CPU...")
    else:
        backend = set_backend("torch", on_error="warn")
        device = "cpu"
        print("Running on CPU...")

    if save_pred:
        y_trn_pred = np.full(y_trn.shape, np.nan)
        for idx, ((train_index, test_index), nest_data_len) in enumerate(zip(cv.split(x_trn),nested_run_onsets)):
            print("Now nested fold predictions...",idx)
            X_train_nest, X_test_nest = x_trn[train_index], x_trn[test_index]
            Y_train_nest, _ = y_trn[train_index], y_trn[test_index]
            nestcv = make_cv(nest_data_len)
            pipeline = make_himalaya_pipeline(n_samples=X_train_nest.shape[0],
                                              n_features=X_train_nest.shape[1],
                                              cv=nestcv,
                                              alpha=alphas,
                                              score_func=correlation_score)
            pipeline.fit(X_train_nest, Y_train_nest)
            pred = pipeline.predict(X_test_nest)
            y_trn_pred[test_index] = backend.to_numpy(pred)
    else:
        y_trn_pred = None

    pipeline = make_himalaya_pipeline(n_samples=x_trn.shape[0],
                                      n_features=x_trn.shape[1],
                                      cv=cv,
                                      alpha=alphas,
                                      score_func=correlation_score)

    pipeline.fit(x_trn, y_trn)
    params = {}
    if  "ridgecv" in pipeline.named_steps:        
        cv_scores = backend.to_numpy(pipeline.named_steps["ridgecv"].cv_scores_)
        params['coef_'] = backend.to_numpy(pipeline.named_steps["ridgecv"].coef_)
    elif "kernelridgecv" in pipeline.named_steps:
        cv_scores = backend.to_numpy(pipeline.named_steps["kernelridgecv"].cv_scores_)
        params['coef_'] = backend.to_numpy(pipeline.named_steps["kernelridgecv"].get_primal_coef())

    cv_scores = backend.to_numpy(cv_scores)
    y_val_pred = pipeline.predict(x_val)
    preds = TrnVal(trn=y_trn_pred, val=y_val_pred)
    score = correlation_score(y_val_pred, y_val)
    score = backend.to_numpy(score)
    print(f"Mean CV score: {cv_scores.mean()}")
    print(f"Mean score: {score.mean()}")

    if perm_type == "gauss":
        rccs = gen_nulldistrib_gauss(len(score), n_samples_val)
    elif perm_type == "block":
        rccs = gen_nulldistrib_block(y_val,
                                     y_val_pred,
                                     device = device)

    significant_voxels, pvalue_corrected = fdr_correction(score, rccs)

    fig = plt.figure()
    plt.hist(score, np.linspace(0, np.max(score), 100), alpha=1.0, label=emb_name)
    plt.title(r"Histogram of correlation coefficient score")
    plt.legend()
    

    return cv_scores, score, pvalue_corrected, fig, params, preds


def multiple_regressor(
    stims: dict[str, TrnVal[np.ndarray]],
    resp: TrnVal[np.ndarray],
    n_iter: int,
    device: int,
    perm_type: str,
    cv_split_points: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], object]:
    """Train an encoder for multiple feature space."""
    alphas = np.logspace(-12, 12, 25)
    cv = make_cv(cv_split_points)
    n_targets_batch = 1000
    n_alphas_batch = 1000
    n_targets_batch_refit = 1000
    solver = "random_search"

    solver_params = dict(
        n_iter=n_iter,
        alphas=alphas,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        score_func=correlation_score
    )

    if device >= 0:
        if device < torch.cuda.device_count():
            torch.cuda.set_device(f"cuda:{device}")
            backend = set_backend("torch_cuda", on_error="warn")
            print("Running on GPU...")

        else:
            print("The CUDA device you specified is not available.")
            print("Running on CPU...")
    else:
        backend = set_backend("torch", on_error="warn")
        print("Running on CPU...")

    y_trn, y_val = resp.trn.astype("float32"), resp.val.astype("float32")
    y_trn, y_val = backend.asarray(y_trn), backend.asarray(y_val)

    xs_trn = []
    xs_val = []
    n_samples_trn = y_trn.shape[0]
    n_features = 0
    n_features_list = []
    for i, (model_name, stim) in enumerate(stims.items()):
        x_trn, x_val = stim.trn, stim.val
        print(f"Shapes of {model_name}: trn={x_trn.shape}, val={x_val.shape}")
        n_features += x_trn.shape[1]
        n_features_list.append(x_trn.shape[1])
        x_trn = x_trn.astype("float32")
        x_val = x_val.astype("float32")
        xs_trn.append(x_trn)
        xs_val.append(x_val)
    x_trn = np.concatenate(xs_trn, 1)
    x_val = np.concatenate(xs_val, 1)

    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]
    print(slices)

    if n_features > n_samples_trn:
        print("Solving Multiple Kernel Ridge regression using random search...")
        ridge = MultipleKernelRidgeCV(
            kernels="precomputed", solver=solver, solver_params=solver_params, cv=cv, random_state=42
        )
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False), Kernelizer(kernel="linear")
        )
        kernelizers_tuples = [
            (name, preprocess_pipeline, slice_)
            for name, slice_ in zip(stims.keys(), slices)
        ]
        column_kernelizer = ColumnKernelizer(kernelizers_tuples)
        pipeline = make_pipeline(
            column_kernelizer,
            ridge,
        )
    else:
        print("Solving Banded Ridge regression using random search...")
        ridge = BandedRidgeCV(
            groups="input", solver=solver, solver_params=solver_params, cv=cv, random_state=42
        )
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
        )
        ct_tuples = [
            (name, preprocess_pipeline, slice_)
            for name, slice_ in zip(stims.keys(), slices)
        ]

        column_transform = ColumnTransformerNoStack(ct_tuples)
        pipeline = make_pipeline(
            column_transform,
            ridge,
        )

    pipeline.fit(x_trn, y_trn)
    y_val_pred = pipeline.predict(x_val)
    y_val_pred = backend.to_numpy(y_val_pred)
    scores = correlation_score(y_val, y_val_pred)
    scores = backend.to_numpy(scores)
    print(f"Mean score: {scores.mean()}")

    y_val_pred_split = pipeline.predict(x_val, split=True)
    y_val_pred_split = backend.to_numpy(y_val_pred_split)
    split_scores = correlation_score_split(y_val, y_val_pred_split)
    split_scores = backend.to_numpy(split_scores)
    print("n_features_space, n_samples_test, n_voxels", y_val_pred_split.shape)
    for score, model_name in zip(split_scores, stims.keys()):
        print(f'Mean score ({model_name})): {score.mean()}')

    all_scores = {}
    all_preds = {}
    all_pvalues_corrected = {}
    n_samples_val = stim.val.shape[0]

    print("all")
    if perm_type == "gauss":
        rccs = gen_nulldistrib_gauss(len(scores), n_samples_val)
    elif perm_type == "block":
        rccs = gen_nulldistrib_block(y_val,
                                     y_val_pred,
                                     device = device)
    significant_voxels, pvalue_corrected = fdr_correction(scores, rccs)
    # scores[significant_voxels == False] = 0
    all_scores["all"] = scores
    all_preds["all"] = y_val_pred
    all_pvalues_corrected["all"] = pvalue_corrected

    params = {}
    try: # Banded Ridge 
        coefs = backend.to_numpy(ridge.coef_)
    except: # Multiple Kernel Ridge
        coefs = backend.to_numpy(ridge.get_primal_coef(column_kernelizer.get_X_fit()))

    fig = plt.figure()
    for i, (score, pred, slice_, model_name) in enumerate(
        zip(split_scores, y_val_pred_split, slices, stims.keys())
    ):
        print(model_name)
        plt.hist(
            score,
            np.linspace(0, np.max(split_scores), 100),
            alpha=0.3,
            label=model_name,
        )
        if perm_type == "gauss":
            rccs = gen_nulldistrib_gauss(len(score), n_samples_val)
        elif perm_type == "block":
            rccs = gen_nulldistrib_block(y_val,
                                         pred,
                                         device = device)
        significant_voxels, pvalue_corrected = fdr_correction(score, rccs)
        # score[significant_voxels == False] = 0
        all_scores[model_name] = score
        all_preds[model_name] = pred
        all_pvalues_corrected[model_name] = pvalue_corrected
        params[model_name] =  coefs[slice_]

    plt.title(r"Histogram of correlation coefficient score split between kernels")
    plt.legend()

    return all_scores, all_pvalues_corrected, all_preds, fig, params

def main(args) -> None:
    """Runs a training session and returns GTZAN predictor object."""
    assert len(args.time_delays)==2, "Set the start time and end time for the time-delay."
    time_delays_savename = "TR" + '_'.join(map(str, args.time_delays)) + "s"
    print(f"Time delays: {args.time_delays} s")

    print("Loading response data...")
    resp, video_names = load_resp_wholevoxels(args.subject_name, args.dataset)
    rearange_indices, cv_split_points,  = make_rearanging_index_for_cv(video_names.trn)
    resp.trn = resp.trn[rearange_indices]

    if len(args.modality_hparams) == 1:
        args.modality_hparams = [args.modality_hparams[0]]
    
    if args.reg_type=="mono":
        assert len(args.feat_names)==1, "When using mono-regressor, specify only one model features."
        feat_name = args.feat_names[0]
        modality_name = args.modalities[0]
        modality_hparams_savename= '_'.join(map(str, args.modality_hparams))
        if "all" in args.select_layers:
            layers_to_process = [
                d for d in os.listdir(f"{data_const.STIM_DIR}/{modality_name}/{modality_hparams_savename}/{feat_name}") if "layer" in d
            ]
        else:
            layers_to_process = [f"layer{l}" for l in args.select_layers]
            
        for layer_name in layers_to_process:
            print(f"Loading {feat_name}'s {layer_name}...")
            layer_path = f"{data_const.STIM_DIR}/{modality_name}/{modality_hparams_savename}/{feat_name}/{layer_name}"
            scores_root = f"./data/encoding/{args.subject_name}/scores"
            scores_save_dir = f"{scores_root}/{time_delays_savename}/{modality_name}/{modality_hparams_savename}/{feat_name}/{layer_name}"
            
            # Check if the scores are already saved. If so, skip this layer.
            if check_saved_score(scores_save_dir, args.reduce_dim, layer_path, args.dataset):
                continue
            
            os.makedirs(scores_save_dir, exist_ok=True)
            stim = load_stim(layer_path, args.reduce_dim, args.time_delays, args.dataset)
            stim.trn = stim.trn[rearange_indices]
        
            print("Training...")
            cv_scores, scores, pvalue_corrected, fig, params, preds = mono_regressor(
                stim, resp, feat_name, args.device, args.perm_type, cv_split_points, args.save_pred
                )
            
            filename = make_filename(args.reduce_dim, args.dataset)
            np.save(f"{scores_save_dir}/cv_cc_{filename}.npy", cv_scores)
            np.save(f"{scores_save_dir}/cc_{filename}.npy", scores)
            np.save(f"{scores_save_dir}/pvalues_corrected_{filename}.npy", pvalue_corrected)
            np.save(f"{scores_save_dir}/coef_{filename}.npy", params['coef_'])
            fig.savefig(f"{scores_save_dir}/dist_{filename}.png")
            if args.save_pred:
                np.save(f"{scores_save_dir}/preds_trn_{filename}.npy", preds.trn)
                np.save(f"{scores_save_dir}/preds_val_{filename}.npy", preds.val)

    else:
        assert len(args.feat_names)>1, "When using multi-regressor, specify two or more models' features."
        assert len(args.modalities)==len(args.modality_hparams)==len(args.feat_names)==len(args.select_layers), "Specify the same number of hyperparameters for modality, model name, and layer, respectively."
        
        scores_root = f"./data/encoding/{args.subject_name}/scores"
        modality_savename= '_'.join(map(str, args.modalities))
        modality_hparams_savename= '_'.join(map(str, args.modality_hparams))
        feat_names_savename = '_'.join(map(str, args.feat_names))
        select_layers_savename = '_'.join(map(str, args.select_layers))
        scores_save_dir = f"{scores_root}/{time_delays_savename}/{modality_savename}/{modality_hparams_savename}/{feat_names_savename}/layer{select_layers_savename}"
        featname_params_layers = f"{modality_savename}/{modality_hparams_savename}/{feat_names_savename}/layer{select_layers_savename}"
        if check_saved_score(scores_save_dir, args.reduce_dim, featname_params_layers, args.dataset):
            return None
        
        all_stim = {}
        for modality, feat_name, modality_hparam, select_layer in zip(args.modalities, args.feat_names, args.modality_hparams, args.select_layers):
            print(f"Loading {modality}-{modality_hparam}-{feat_name}-layer{select_layer}'s embeddings...")
            layer_path = f"{data_const.STIM_DIR}/{modality}/{modality_hparam}/{feat_name}/layer{select_layer}"
            stim = load_stim(layer_path, args.reduce_dim, args.time_delays, args.dataset)
            stim.trn = stim.trn[rearange_indices]
            all_stim[modality+"_"+modality_hparam+"_"+feat_name+"_"+select_layer] = stim

        print("Training...")
        all_scores, all_pvalues_corrected, all_preds, fig, params = multiple_regressor(
            all_stim, resp, args.n_iter, args.device, args.perm_type, cv_split_points)

        os.makedirs(scores_save_dir, exist_ok=True)
        filename = make_filename(args.reduce_dim, args.dataset)
        np.save(f"{scores_save_dir}/cc_{filename}.npy", all_scores)
        np.save(f"{scores_save_dir}/pvalues_corrected_{filename}.npy", all_pvalues_corrected)
        np.save(f"{scores_save_dir}/preds_{filename}.npy", all_preds)
        np.save(f"{scores_save_dir}/coef_{filename}.npy", params)
        fig.savefig(f"{scores_save_dir}/dist_{filename}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a decoding model to predict embedding from fmri data."
    )

    parser.add_argument(
        "--subject_name",
        type=str,
        required=True,
        help="Name of the subject to train the model on.",
    )

    parser.add_argument(
        "--time_delays",
        nargs="*",
        type = int,
        required=True,
        default=[3, 5],
        help="Number of time delays. Set the start time and end time for the time-delay",
    )
    
    parser.add_argument(
        "--modalities",
        nargs="*",
        type=str,
        required=True,
        help="Name of the modality to use."
    )
    
    parser.add_argument(
        "--modality_hparams",
        nargs="*",
        type=str,
        required=True,
        default="default",
        help="Specific modality's hparams."
    )

    parser.add_argument(
        "--feat_names",
        nargs="*",
        type=str,
        required=True,
        help="Names of the feature to use.",
    )
    
    parser.add_argument(
        "--select_layers",
        nargs="*",
        type=str,
        required=True,
        default="all",
        help="Number of layer to use. Set None if there is no layer.",
    )

    parser.add_argument(
        "--device",
        type=int,
        required=True,
        default=0,
        help="GPU number",
    )

    parser.add_argument(
        "--reg_type",
        choices=["mono", "multi"],
        required=True,
        help="Type of the regressor.",
    )

    parser.add_argument(
        "--perm_type",
        choices=["gauss", "block"],
        required=True,
        help="Type of the significance testing.",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
        required=False,
        help="Number of random search interations.",
    )
    
    parser.add_argument(
        "--reduce_dim",
        nargs="*",
        type=str,
        default = ["default", None],
        required=False,
        help="Dimension reduction method and its hyperparameter.",
    )

    parser.add_argument(
        "--save_pred",
        action='store_true',
        help="for saving cross-validated prediction of training dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "splitA", "splitB"],
        default="all",
        required=False,
        help="Dataset to use. Select splitA and splitB for quality control analysis of drama dataset."
    )


    main(parser.parse_args())
