import numpy as np
from drama2brain import data_const
from drama2brain.utils import expand_HRF_stim, fit_pca, fit_srp
from drama2brain.utils import TrnVal
import os
import pickle

def load_feature_video(video_name,
                       feat_path,
                       trainvalid,
                       trainvalid_frames,
                       return_part_lengths=False):
    features = []
    part_lengths = []
    for part, trainvalid_frame in enumerate(trainvalid_frames[video_name]):
        if trainvalid_frame == trainvalid:
            datpath = f"{feat_path}/{video_name}_{part+1:03d}.npy" # Filename is 1-based
            data = np.load(datpath)
            if part+1 > 1: # Filename is 1-based
                data = data[20:] # Exclude overlapped frame

            print(f"Loaded data: {video_name}_{part+1}, feat_name:{feat_path} datashape {data.shape} ")
            features.append(data.reshape(data.shape[0],-1))
            part_lengths.append(data.shape[0])
            
    if len(features)>0:
        features = np.vstack(features)
        
    if return_part_lengths:
        return features, part_lengths
    else:
        return features


def collect_stim(feat_path,
                 trainvalid,
                 ishrfexpand,
                 time_delays,
                 dataset):

    if dataset == "all":
        trainvalid_frames = data_const.TRAINVALID_FRAMES
    elif dataset == "splitA":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_A
    elif dataset == "splitB":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_B

    features = []
    for video_name in data_const.VIDEO_NAMES:
        features_cvideo = load_feature_video(video_name,
                                             feat_path,
                                             trainvalid,
                                             trainvalid_frames)
        if len(features_cvideo)==0:
            continue
        if ishrfexpand:
            features_cvideo = expand_HRF_stim(features_cvideo,
                                              mode="avg",
                                              start=time_delays[0],
                                              end=time_delays[1])
        features.append(features_cvideo)
            
    features = np.vstack(features)

    return features


def save_features_splited_by_video(features, lengths, video_names, trainvalid, reduce_dim, feat_path, trainvalid_frames, ishrfexpand=True, time_delays=[3, 5]):
    idx = 0
    hrfexpanded_features_all = []
    for video_name, part_lengths in zip(video_names[trainvalid], lengths[trainvalid]):
        if trainvalid == "VALID":
            npart_trn = trainvalid_frames[video_name].count("TRAIN")
            part = npart_trn + 1
        else:
            part = 1
        hrfexpanded_features = []
        print(video_name)
        for length in part_lengths:
            print(part, length, features.shape)
            part_feature = features[idx: idx + length]
            if reduce_dim[0]=="pca":
                part_file = f"{feat_path}/{video_name}_{part:03d}_{reduce_dim[1]}PCs.npy"
            elif reduce_dim[0]=="srp":
                part_file = f"{feat_path}/{video_name}_{part:03d}_{reduce_dim[1]}eps.npy"
            np.save(part_file, part_feature)
            idx += length
            part += 1
            if ishrfexpand:
                part_feature_hrfexpanded = expand_HRF_stim(part_feature,
                                                           mode="avg",
                                                           start=time_delays[0],
                                                           end=time_delays[1])
                hrfexpanded_features.append(part_feature_hrfexpanded)
        hrfexpanded_features = np.vstack(hrfexpanded_features)
        hrfexpanded_features_all.append(hrfexpanded_features)
    hrfexpanded_features_all = np.vstack(hrfexpanded_features_all)
    
    return hrfexpanded_features_all
                

def check_and_load_reduced_features(feat_path, reduce_dim, trainvalid_frames, ishrfexpand=True, time_delays=[3, 5]):
    train_features_all = []
    valid_features_all = []
    for video_name in data_const.VIDEO_NAMES:
        print(video_name)
        train_features = []
        valid_features = []
        for part, trainvalid_frame in enumerate(trainvalid_frames[video_name]):
            if reduce_dim[0] == "pca":
                part_file = f"{feat_path}/{video_name}_{part+1:03d}_{reduce_dim[1]}PCs.npy"
            elif reduce_dim[0] == "srp":
                part_file = f"{feat_path}/{video_name}_{part+1:03d}_{reduce_dim[1]}eps.npy"
            
            # Check if dim-reduced features already exist. If so, load the file.
            if os.path.exists(part_file):
                part_feature = np.load(part_file)
                if ishrfexpand:
                    part_feature = expand_HRF_stim(part_feature,
                                                    mode="avg",
                                                    start=time_delays[0],
                                                    end=time_delays[1])
                if trainvalid_frame == "TRAIN":
                    train_features.append(part_feature)
                elif trainvalid_frame == "VALID":
                    valid_features.append(part_feature)
            else:
                return None

        if len(train_features)>0:
            train_features = np.vstack(train_features)
            train_features_all.append(train_features)
        if len(valid_features)>0:
            valid_features = np.vstack(valid_features)
            valid_features_all.append(valid_features)
            
    print(f"{feat_path}'s reduced features' already exist.")
    print(f"Loading this saved file...")
    train_features_all = np.vstack(train_features_all)
    valid_features_all= np.vstack(valid_features_all)
    features = TrnVal(trn=train_features_all, val=valid_features_all)
    print(f"Reduced features' shape: {features.trn.shape}, {features.val.shape}")
        
    return features
  

def collect_stim_for_dimreducer(
    feat_path,
    reduce_dim,
    ishrfexpand,
    time_delays,
    dataset
    ):
    
    if reduce_dim[0] == "pca":
        reduce_dim[1] = int(reduce_dim[1])
    elif reduce_dim[0] == "srp":
        reduce_dim[1] = float(reduce_dim[1])

    if dataset == "all":
        trainvalid_frames = data_const.TRAINVALID_FRAMES
    elif dataset == "splitA":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_A
    elif dataset == "splitB":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_B
    
    stim_features = check_and_load_reduced_features(feat_path, reduce_dim, trainvalid_frames, ishrfexpand, time_delays)
    if stim_features is not None:
        return stim_features

    # Collect features
    print(f"Reduced features don't exist. Collecting features...")
    features = {trainvalid: [] for trainvalid in ["TRAIN", "VALID"]}
    lengths = {trainvalid: [] for trainvalid in ["TRAIN", "VALID"]}
    video_names = {trainvalid: [] for trainvalid in ["TRAIN", "VALID"]}
    for trainvalid in ["TRAIN", "VALID"]:
        for video_name in data_const.VIDEO_NAMES:
            print(video_name)
            features_cvideo, part_length = load_feature_video(
                video_name,
                feat_path,
                trainvalid,
                trainvalid_frames,
                return_part_lengths=True
                )
            if len(features_cvideo) == 0:
                continue
            features[trainvalid].append(features_cvideo)
            lengths[trainvalid].append(part_length)
            video_names[trainvalid].append(video_name)
            
        features[trainvalid] = np.vstack(features[trainvalid])
        
    features = TrnVal(trn=features["TRAIN"], val=features["VALID"])

    print(f"Reducing features...")
    print(f"Original features' shape: {features.trn.shape}, {features.val.shape}")
    if reduce_dim[0]=="pca":
        features, projector = fit_pca(features, reduce_dim[1], return_projector=True)
        try:
            np.save(f"{feat_path}/projector_{reduce_dim[1]}PCs.npy", projector)
        # When the size of the projector is too large, save it as a pickle file.
        except:
            with open(f"{feat_path}/projector_{reduce_dim[1]}PCs.pkl", 'wb') as f:
                pickle.dump(projector, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif reduce_dim[0]=="srp":
        features, projector = fit_srp(features, reduce_dim[1], return_projector=True)
        try:
            np.save(f"{feat_path}/projector_{reduce_dim[1]}eps.npy", projector)
        except:
            with open(f"{feat_path}/projector_{reduce_dim[1]}eps.pkl", 'wb') as f:
                pickle.dump(projector, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Reduced features' shape: {features.trn.shape}, {features.val.shape}")
    features_trn = save_features_splited_by_video(features.trn, lengths, video_names, "TRAIN", reduce_dim, feat_path, trainvalid_frames, ishrfexpand, time_delays)
    features_val = save_features_splited_by_video(features.val, lengths, video_names, "VALID", reduce_dim, feat_path, trainvalid_frames, ishrfexpand, time_delays)
    features = TrnVal(trn=features_trn, val=features_val)
    
    return features

"""
python -m drama2brain.stim_loader
"""
if __name__ == "__main__":
    # Test
    feat_path = ""
    reduce_dim = ["pca", 1280]
    features = collect_stim_for_dimreducer(
        feat_path, reduce_dim, ishrfexpand=True, time_delays=[3, 5], dataset="all")
    print(features.trn.shape, features.val.shape)