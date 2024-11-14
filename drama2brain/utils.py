import numpy as np
import os
from PIL import Image
import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

import dataclasses
from typing import Callable, Generic, TypeVar
from drama2brain import data_const
from statsmodels.stats.multitest import fdrcorrection
from himalaya.scoring import correlation_score
import torch
from sklearn.model_selection import check_cv
from himalaya.kernel_ridge import (
    KernelRidgeCV,
)
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")
U = TypeVar("U")
@dataclasses.dataclass
class TrnVal(Generic[T]):
    """Tuple of something existing for training and evaluation data.

    This may be a trained model, a path, or data, for example."""

    trn: T
    val: T

    def map_fn(self, f: Callable[[T], U]) -> "TrnVal[U]":
        """Applies f to trn and val."""
        return TrnVal(trn=f(self.trn), val=f(self.val))

def expand_HRF_resp(data,
                    mode):
    # print(f"Original Data size = {data.shape}")
    shifts = [-3, -4, -5] # Rolling up response matrix
    shifted_datas = []
    for s in shifts:
        shifted_data = np.roll(data, shift=s, axis=0)
        shifted_datas.append(shifted_data)
    if mode == "concat":
        data = np.hstack(shifted_datas)
    elif mode == "avg":
        data = np.mean(shifted_datas,axis=0)
    
    # print(f"Reshaped Data size = {data.shape}, mode = {mode}")    
    
    return data

def expand_HRF_stim(data, mode, start, end):
    # print(f"Original Data size = {data.shape}")
    shifts = [i for i in range(start, end+1)] # Rolling down stimulus matrix
    zeros = np.zeros((end,data.shape[1]))
    data_new = np.concatenate((data,zeros),axis=0)
    shifted_datas = []
    for s in shifts:
        shifted_data = np.roll(data_new, shift=s, axis=0)
        shifted_datas.append(shifted_data)
    if mode == "concat":
        data_new = np.hstack(shifted_datas)
    elif mode == "avg":
        data_new = np.mean(shifted_datas,axis=0)
    data_new = data_new[:data.shape[0],:]
    
    # print(f"Reshaped Data size = {data.shape}, mode = {mode}")    
    
    return data_new

def resize_image(file_path, new_file_path, width, height):
    with Image.open(file_path) as img:
        resized_img = img.resize((width, height))
        resized_img.save(new_file_path)

def resize_images(source_dir, target_dir, width, height):
    """
    Resize all images in a directory to the specified width and height,
    and save them in a new directory structure mirroring the original.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    for root, dirs, files in os.walk(target_dir):
        if files:
            print("Target directory already exists.")
            return
    
    print("Resizing images...")
    os.makedirs(target_dir, exist_ok=True)

    tasks = []
    with ThreadPoolExecutor() as executor:
        for root, dirs, files in os.walk(source_dir):
            # Creating corresponding directories in the target directory
            for dir in dirs:
                new_dir_path = os.path.join(root, dir).replace(source_dir, target_dir)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)

            # Resizing images and saving them in the corresponding location
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    new_file_path = file_path.replace(source_dir, target_dir)
                    tasks.append(executor.submit(resize_image, file_path, new_file_path, width, height))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

def load_frames(frames_topdir):
    frames_dirs = [dir_path for dir_path in glob.glob(os.path.join(frames_topdir, "*")) if os.path.isdir(dir_path)]
    frames_dirs = sorted(frames_dirs)
    frames_dict = {}

    for dir_path in frames_dirs:
        dir_name = os.path.basename(dir_path)        
        img_paths = glob.glob(os.path.join(dir_path, "*.png"))
        frames_dict[dir_name] = sorted(img_paths)
        print(f"{len(img_paths)} images in {dir_path}")

    return frames_dict

def load_captions(movname, annot_type):
    captiondir = data_const.CAPTION_DIR
    captions = []

    def convert_filename(movname, annot_type):
        series_name = movname.split('_')[0]
        run_number = int(movname.split('_')[-1])
        new_movname = f"{series_name}_run{run_number}_{annot_type}_each1sec"
        return new_movname

    new_movname_en = f"{captiondir}/{convert_filename(movname, annot_type)}_en.txt"

    if os.path.exists(new_movname_en):
        captions = pd.read_csv(new_movname_en, encoding="utf-8",sep="\t").values.tolist()
    else:
        Exception("Can't find:", new_movname_en)

    return captions

def downsample(frames):
    frames_ds = []

    fps = 30
    for second in range(len(frames) // fps):
        start_index = second*fps
        middle_index = second*fps + fps//2
        frames_ds.append(frames[start_index])
        frames_ds.append(frames[middle_index])
    frames_ds = np.array(frames_ds)

    return frames_ds

def fit_pca(stim, num_pcs, return_projector=False):
    rng = 42
    pca = PCA(random_state=rng, n_components=num_pcs)
    projector = pca.fit(stim.trn)
    stim_trn_pca = projector.transform(stim.trn)
    stim_val_pca = projector.transform(stim.val)
    Stim = TrnVal(trn=stim_trn_pca, val=stim_val_pca)
    if return_projector:
        return Stim, projector
    else:
        return Stim

def fit_srp(stim, epsilon, return_projector=False):
    """
    Fit and transform using Sparse Random Projection
    The number of dimensions after fit is determined by the number of samples at the time of fitting and the value of epsilon
    Example 1: If you fit with 22,200 samples and set epsilon to 0.1, the number of dimensions after transformation will be 8,578.
    Example 2: If you fit with 22,200 samples and set epsilon to 0.2, the number of dimensions after transformation will be 2,309.
    """
    rng = 42
    srp = SparseRandomProjection(random_state=rng,eps=epsilon)
    projector = srp.fit(stim.trn)
    stim_trn_srp = projector.transform(stim.trn)
    stim_val_srp = projector.transform(stim.val)
    Stim = TrnVal(trn=stim_trn_srp, val=stim_val_srp)
    if return_projector:
        return Stim, projector
    else:
        return Stim

def gen_nulldistrib_gauss(nvoxels: int, 
                          valnum: int) -> list[np.ndarray]:
    rccs = []

    # Max num of cortex voxels = 400 x 400
    a = np.random.randn(400, valnum)
    b = np.random.randn(400, valnum)
    rccs = np.corrcoef(a, b)
    rccs = rccs[400:, :400].ravel()
    rccs = rccs[:nvoxels]

    return rccs

def gen_nulldistrib_block(resp_true: torch.Tensor,
                          resp_pred: np.ndarray,
                          block_size = 10,
                          num_iterations = 1,
                          device = 0) -> list[np.ndarray]:
    """
    Block permutation: c.f. Tang et al., 2023 NeurIPS https://arxiv.org/abs/2305.12248
    Note that the implementation slightly differs from Tang et al. for computational reasons:
    - This function performs pure block permutation without replacement.
    - The default value of num_iterations is set to 1, so not creating a null distribution for each voxel.
    """
    
    np.random.seed(42)
    num_trials = resp_true.shape[0]
    last_block_size = num_trials % block_size
    rccs = []
    resp_pred = torch.Tensor(resp_pred).to(device)
    for i in range(num_iterations):
        if last_block_size == 0:
            indices = np.arange(num_trials).reshape(-1, block_size).tolist()
        else:
            l1 = np.arange(num_trials - last_block_size).reshape(-1, block_size).tolist()
            l2 = [np.arange(num_trials - last_block_size, num_trials).tolist()]
            indices = l1 + l2

        np.random.shuffle(indices)
        shuffled_indices = np.concatenate(indices)
        resp_true_shuffle = resp_true[shuffled_indices, :]
        # Compute linear correlation
        rcc = correlation_score(resp_pred, resp_true_shuffle)
        rccs.extend(rcc.detach().cpu().numpy())

    return rccs

def fdr_correction(ccs: np.ndarray,
                   rccs: np.ndarray,
                   ) -> np.ndarray:
    # Make random correlation coefficient histogram
    nvoxels = len(ccs)

    px = []
    for i in range(nvoxels):
        x = np.argwhere(rccs > ccs[i])
        px.append(len(x) / nvoxels)

    significant_voxels, pvalue_corrected = fdrcorrection(px, alpha=0.05, method="indep", is_sorted=False)
    if sum(significant_voxels) > 0:
        print(f"Minimum R of siginificant voxels = {np.min(ccs[significant_voxels])}")
        print(f"Maximum R of siginificant voxels = {np.max(ccs[significant_voxels])}")
        print(
            f"Number of voxels with significant positive correlation: {len(np.where(significant_voxels)[0])}"
        )
    else:
        print("No voxels are significant")

    return significant_voxels, pvalue_corrected

def bootstrap_uniquevariance(pipelines, 
                             resp, 
                             stim,
                             num_iterations=1000):
    """
    Perform bootstrapping analysis to evaluate the significance of unique variance explained by each feature subset.
    Henderson et al., 2023 JNS https://www.jneurosci.org/content/43/22/4144

    :param pipeline: himalaya pipelines: pipeline[0] is full-model, pipelines[n] where n>0 is leave-one-model-out-model
    :param resp: Array of voxel responses (n_samples, n_voxels).
    :param stim: Array of stimuli (n_samples, n_dimensions).
    :param n_iterations: Number of bootstrap iterations (default 1000).
    :return: significant_voxels for each model
    :        pvalues_corrected for each model
    """
    n_samples = resp.shape[0]
    n_voxels = resp.shape[1]
    n_models = len(pipelines)

    # Initialize arrays to store correlation coefficients
    R_full_model = np.zeros((num_iterations, n_voxels))
    R_partial_models = np.zeros((num_iterations, n_voxels, n_models))

    # Perform bootstrap iterations
    for i in range(num_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_voxel_data = resp[indices]
        resampled_stim = stim[indices]

        # Compute R for full model
        predictions = pipelines[0].predict(resampled_stim)
        R_full_model[i,:] = correlation_score(resampled_voxel_data, predictions).detach().cpu().numpy()

        # Compute R for each partial model
        for j in range(n_models):
            predictions = pipelines[j].predict(resampled_stim, split = True)
            R_partial_models[i, :, j] = correlation_score(resampled_voxel_data, predictions[j]).detach().cpu().numpy()

    # Compute R_unique (n_iterations x n_voxels x n_models - 1)
    R_unique = R_full_model[:, :, np.newaxis] - R_partial_models
    count_Runique_le_zero = np.sum(R_unique <= 0, axis=0)

    # Calculate p_value r=<0
    p_values = count_Runique_le_zero / num_iterations # n_voxels x n_models

    # FDR correction
    pvalues_corrected = []
    significant_voxels = []
    for i in range(n_models - 1):
        sig, p = fdrcorrection(p_values[:,i], alpha=0.05, method="indep", is_sorted=False)
        pvalues_corrected.append(p)
        significant_voxels.append(sig)

    return significant_voxels, pvalues_corrected

def make_rearanging_index_for_cv(video_names):
    """
    Make rearanging indices by data_const.VIDEO_CVIDX for stratified (i.e. non-random) cross-validation.
    This is important because our stimulus is continuous, thus having temporal correlation across frames.
    This function also makes cv_split_points for making a split generator.
    """

    # Make rearange indices
    cv_indices = np.array([data_const.VIDEO_CVIDX[video_name] for video_name in video_names])
    rearange_indices = np.argsort(cv_indices)

    # Make split points for cross-validation split generator
    cv_indices = cv_indices[rearange_indices]
    ncv = np.max(cv_indices) + 1
    cv_split_points = [np.sum(cv_indices==i) for i in range(ncv)]
    
    return rearange_indices, cv_split_points

def generate_leave_one_run_out(n_samples, run_onsets):
    """Generate a leave-one-run-out split for cross-validation.
    Adapted from Gallant lab's voxel-wise modeling tutorial
    https://gallantlab.org/voxelwise_tutorials/_auto_examples/shortclips/03_plot_wordnet_model.html#define-the-cross-validation-scheme
    
    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
        
    """

    n_runs = len(run_onsets)
    all_val_runs = np.array(
        [[i for i in range(n_runs)]])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val

def make_cv(data_lens,
            return_nested_run_onsets=False):
    """
    This function makes non-random CV index generator.
    """


    n_samples_train = sum(data_lens)
    run_onsets = [0] + np.cumsum(data_lens).tolist()[:-1]
    cv = generate_leave_one_run_out(n_samples_train,
                                    run_onsets)
    cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list
    if return_nested_run_onsets:
        nested_run_onsets = []
        for i in range(0,len(data_lens)):
            nested_len = data_lens.copy()
            import pdb;pdb.set_trace()
            nested_len.pop(i)
            nested_run_onsets.append(nested_len)
        return cv, nested_run_onsets
    else:
        return cv 

def make_himalaya_pipeline(n_samples,
                           n_features,
                           cv,
                           alpha,
                           score_func):
    if n_samples >= n_features:
        print("Solving Ridge regression...")
        ridge = RidgeCV(
            alphas=alpha, cv=cv, solver_params={"score_func": score_func,
                                                "n_targets_batch":3000,
                                                "n_alphas_batch":20,
                                                "n_targets_batch_refit":200,
            }
        )

    else:
        print("Solving Kernel Ridge regression...")
        ridge = KernelRidgeCV(
            alphas=alpha, cv=cv, solver_params={"score_func": score_func,
                                                "n_targets_batch":3000,
                                                "n_alphas_batch":20,
                                                "n_targets_batch_refit":200,
            }
        )
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )

    return pipeline