from collections import defaultdict
import numpy as np
import os, json, h5py
import scipy.io
from drama2brain import data_const
from drama2brain.utils import expand_HRF_resp

def load_fs_roi(subject_name: str) -> dict:
    return scipy.io.loadmat(f"{data_const.FMRI_BBREGISTER_DIR}//{subject_name}/fsROI.mat")

def load_vset(subject_name: str) -> dict:
    return scipy.io.loadmat(f"{data_const.FMRI_BBREGISTER_DIR}//{subject_name}/vset_099.mat")


def prepare_roi(subject_name):
    fs_roi = load_fs_roi(subject_name)

    labels = fs_roi["fsROI"][0][0][1]
    labels = [l[0][0] for l in labels]

    voxels = fs_roi["fsROI"][0][0][2]
    voxels = [v[0][:, 0] for v in voxels]

    # Cerebral Cortex voxels
    vset = load_vset(subject_name)
    tvoxels = vset["tvoxels"] - 1
    voxels = [v[0][:, 0] - 1 for v in voxels]

    # Use only cortex
    ctx_voxels = []
    for l, v in zip(labels, voxels):
        if l[:3] == ("ctx"):
            ind = np.where(np.in1d(tvoxels, v) == 1)[0]
            ctx_voxels.append((l, ind))

    return ctx_voxels, tvoxels

def get_fmridata_dict(subject_name):
    expfile = f"{data_const.FMRI_EXP2019_DIR}/subject.json"
    with open(expfile, 'r') as file:
        expfile = json.load(file)
        expfile = expfile[data_const.SUBJ_NAME.index(subject_name)]
        movdata = expfile["mov"]

    fmridata_dict = defaultdict(list)
    
    for mov in movdata:
        if mov['title'] in data_const.VIDEO_NAMES:
            for run in mov['run']:
                dirname = run['file'].split('/')[2]
                filename = run['file'].split('/')[3]
                fmridata_dict[mov["title"]].append(dirname+"/"+filename)

    return fmridata_dict

def load_fmri_video(video_name,
                    trainvalid,
                    tvoxels,
                    subject_name,
                    isaddlastframes,
                    trainvalid_frames):
    
    data_dir = f"{data_const.FMRI_RAW_DIR}/{subject_name}/"    
    save_dir = f"{data_const.FMRI_PRE_DIR}/{subject_name}/"
    os.makedirs(save_dir, exist_ok=True)

    fmridata_dict = get_fmridata_dict(subject_name)
    npart = trainvalid_frames[video_name].count(trainvalid)
    
    data = []
    vidcount = 0
    for vididx, (fname_part, trainvalid_frame) in enumerate(zip(fmridata_dict[video_name], trainvalid_frames[video_name])):
        if trainvalid_frame == trainvalid:
            fname_video = f"{save_dir}/{video_name}_{vididx}_CerebCortex.npy"

            if os.path.exists(fname_video):
                dataDT = np.load(fname_video)
                print(f"Loading: {subject_name}, {fname_video}, shape= {dataDT.shape}")
            else:
                print(f"Preprocessing data: {subject_name}, {fname_video} part {vididx}")
                f = h5py.File(data_dir+fname_part,'r')
                dataDT = np.squeeze(np.array(f['dataDT'])).astype("float32")
                dataDT = dataDT[:,tvoxels].squeeze()
                print(f"Saving... {fname_video}, shape = {dataDT.shape}")
                np.save(fname_video,dataDT)
            if vidcount == npart - 1 and isaddlastframes:
                if dataDT.shape[0] < data_const.VIDEO_FRAMES[video_name][vididx]+6:
                    import pdb
                    pdb.set_trace()                                    
                dataDT = dataDT[:data_const.VIDEO_FRAMES[video_name][vididx]+6]
            else:
                dataDT = dataDT[:data_const.VIDEO_FRAMES[video_name][vididx]]
            print(f"Discard last frames: {video_name}, {vididx}, {fname_part}, shape = {dataDT.shape}")
            if vididx > 0:
                dataDT = dataDT[20:]
                print(f"Discard first 20 frames: shape = {dataDT.shape}")
            print(f"Loaded: {subject_name}, {fname_video}, shape= {dataDT.shape}")
            data.append(dataDT)
            vidcount += 1
    data = np.vstack(data).squeeze()

    return data

def collect_fmri_byroi(subject_name,
                       trainvalid,
                       mode):

    ctx_voxels, tvoxels = prepare_roi(subject_name)
    allroidata = defaultdict(list)    
    for video_name in data_const.VIDEO_NAMES:    
        npart = data_const.TRAINVALID_FRAMES[video_name].count(trainvalid)
        if npart == 0:
            continue
        data = load_fmri_video(video_name,
                               trainvalid,
                               tvoxels,
                               subject_name,
                               isaddlastframes=True)

        for label, voxels in ctx_voxels:
            croidata = expand_HRF_resp(data[:,voxels], mode)[:-6,:]
            allroidata[label].append(croidata)
        print(f"Video {video_name} shape = {croidata.shape}")


    for roiname in allroidata.keys():
        allroidata[roiname] = np.vstack(allroidata[roiname])

    return allroidata

def collect_fmri_wholevoxels(subject_name,
                             trainvalid,
                             dataset):
    
    if dataset == "all":
        trainvalid_frames = data_const.TRAINVALID_FRAMES
    elif dataset == "splitA":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_A
    elif dataset == "splitB":
        trainvalid_frames = data_const.TRAINVALID_FRAMES_DATASPLIT_B

    _, tvoxels = prepare_roi(subject_name)
    all_data = []
    all_video_names = []
    for video_name in data_const.VIDEO_NAMES:    
        data = []
        npart = trainvalid_frames[video_name].count(trainvalid)
        if npart == 0:
            continue
        data = load_fmri_video(video_name,
                               trainvalid,
                               tvoxels,
                               subject_name,
                               isaddlastframes=False,
                               trainvalid_frames=trainvalid_frames)

        all_data.append(data)
        all_video_names += [video_name]*len(data)

        print(f"Video {video_name} shape = {data.shape}")

    all_data = np.vstack(all_data)
    all_video_names = np.array(all_video_names)
    print(f"ALL Video shape = {all_data.shape}")

    return all_data, all_video_names

if __name__ == "__main__":
    subject_name = "DM01"
    trainvalid = "TRAIN"
    mode = "avg"
    allroidata = collect_fmri_byroi(subject_name,
                                    trainvalid,
                                    mode)
    for i,v in allroidata.items():
        print(i, v.shape)
        break
    
