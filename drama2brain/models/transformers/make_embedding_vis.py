from tqdm import tqdm
from drama2brain.models.transformers.model_const_vis import model_dict_vision
from drama2brain.utils import resize_images, load_frames, downsample
import os
import numpy as np
import torch
from PIL import Image
from drama2brain.data_const import FRAME_DIR
import torch.nn as nn
import gc


def embedding_maker_vis(devices):
    model_names = model_dict_vision
    if devices[0] >= 0:
        first_device = f"cuda:{devices[0]}"
    else:
        first_device = f"cpu"

    for model_name, (processor, model, model_path) in model_names.items():
        print(model_name)
        emb_save_path = os.path.join("./data/stim_features/vision", model_name)
        os.makedirs(emb_save_path, exist_ok=True)
        
        if model_name == "CLIP":
            saved_model_path = f"./data/model_ckpts/vision-semantic/{model_path}"
            if first_device == "cpu":
                model = model.from_pretrained(saved_model_path, trust_remote_code=True)
            else:
                model = model.from_pretrained(saved_model_path, torch_dtype=torch.float16, trust_remote_code=True)
            model = model.vision_tower
        else:
            saved_model_path = f"./data/model_ckpts/vision/{model_path}"
            model = model.from_pretrained(saved_model_path)
        if len(devices) > 1:
            model = nn.DataParallel(model, device_ids=devices)
        model = model.to(first_device)
        processor = processor.from_pretrained(saved_model_path)
        try:
            size = model.config.image_size
        except:
            size = 224
        print(f"Image size: {size}")
        source_directory = f'{FRAME_DIR}/frames'
        
        if type(size) == list:
            width, height = size[0], size[1]
        else:
            width, height = size, size
            
        target_directory = f'{FRAME_DIR}/frames_{width}x{height}px'
        resize_images(source_directory, target_directory, width, height)
        
        frames_all = load_frames(target_directory)
        # Iterate all videos
        for movname, frame_paths in frames_all.items():
            if os.path.exists(f"{emb_save_path}/{movname}.npy"):
                print(f"Already exist: {movname} with {len(frame_paths)} frames")
                continue
            print(f"Now processing: {movname} with {len(frame_paths)} frames")
            frames_ds = downsample(frame_paths)
            embs_all = convert_image_to_embedding(frames_ds, processor, model_name, model, devices, first_device)
            for layer, embs in embs_all.items():
                print(f"{model_name} {layer}'s embedding shape: {embs.shape}")
                layer_save_path = f"{emb_save_path}/{layer}"
                os.makedirs(layer_save_path, exist_ok=True)
                np.save(f"{layer_save_path}/{movname}.npy", embs)
            

def convert_image_to_embedding(frames_ds, image_processor, model_name, model, devices, first_device):
    embs_all = {}
    for frame_idx, frame_path in enumerate(tqdm(frames_ds)):
        img = Image.open(frame_path)
        if model_name == "CLIP":
            prompt = "USER: <image>\nTest\nASSISTANT:"
            input = image_processor(prompt, img, return_tensors='pt')
            if first_device == "cpu":
                input = input["pixel_values"]
            else:
                input = input["pixel_values"].to(torch.float16)
        else:
            input = image_processor(img, return_tensors="pt")
        with torch.no_grad():
            if len(devices) == 1:
                if devices[0] >= 0:
                    input = input.to(f"cuda:{devices[0]}")
                else:
                    input = input.to("cpu")
            if model_name == "CLIP":
                embs = model(input, output_hidden_states=True)
            else:
                embs = model(**input, output_hidden_states=True)
            embs = embs.hidden_states
            if frame_idx==0:
                embs_all = {f"layer{layer_idx+1}":[] for layer_idx in range(len(embs))}
                
            for layer_idx, emb in enumerate(embs):
                emb = emb.flatten()
                emb = emb.cpu().detach().numpy()
                embs_all[f"layer{layer_idx+1}"].append(emb)
        
            del input, embs, emb, img
            gc.collect()
            torch.cuda.empty_cache()

    embs_all = {layer:np.array(embs) for layer, embs in embs_all.items()}
    # Averaging by 2 frames
    embs_all = {layer:embs.reshape(-1, 2, embs.shape[-1]).mean(axis=1) for layer, embs in embs_all.items()}
   
    return embs_all
