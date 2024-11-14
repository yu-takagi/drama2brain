from tqdm import tqdm
from drama2brain.models.transformers.model_const_vis_sem import model_dict_vis_sem
from drama2brain.utils import resize_images, load_frames, downsample, load_captions
import os
import numpy as np
import torch
from PIL import Image
from drama2brain.data_const import FRAME_DIR
import torch.nn as nn


def embedding_maker_vis_sem(devices, hparam, token_type):
    print(hparam)
    vision_hparam, semantic_hparam = hparam.split("-")
    model_names = model_dict_vis_sem
    if devices[0] < 0:
        first_device = "cpu"
    else:
        first_device = f"cuda:{devices[0]}"

    for model_name, (processor, model, model_path) in model_names.items():
        print(model_name)

        saved_model_path = f"./data/model_ckpts/vision-semantic/{model_path}"
        saved_processor_path = f"./data/model_ckpts/vision-semantic/{model_path}"
        model = model.from_pretrained(saved_model_path, torch_dtype=torch.float16, trust_remote_code=True)
        
        if model_name in ["Kosmos-2"]:
            processor = processor.from_pretrained(saved_processor_path, local_files_only=True)
        else:
            processor = processor.from_pretrained(saved_processor_path, trust_remote_code=True)

        if model_name in ["GIT"]:
            size = list(processor.image_processor.crop_size.values())
            patch_size = model.config.vision_config.patch_size
        elif model_name in ["LLaVA-v1.5-question-inputs", "LLaVA-v1.5", "BridgeTower", "Kosmos-2"]:
            size = model.config.vision_config.image_size
            patch_size = model.config.vision_config.patch_size
        
        if type(size) == list:
            width, height = size[0], size[1]
        else:
            width, height = size, size
        
        if model_name == "Kosmos-2":
            vis_token_num = 68
        else:
            vis_token_num = int(width * height  / patch_size / patch_size)
                    
        print(f"Image size: {size}")
        print(f"Number of Vision tokens: {vis_token_num}")
        
        if len(devices) > 1:
            model = nn.DataParallel(model, device_ids=devices)
        model = model.to(first_device)
        
        source_directory = f'{FRAME_DIR}/frames'
        target_directory = f'{FRAME_DIR}/frames_{width}x{height}px'
        resize_images(source_directory, target_directory, width, height)
        frames_all = load_frames(target_directory)
        
        if token_type == "avgToken":
            print("Token type: Average token embeddings")
            emb_save_path = f"./data/stim_features/vision-semantic/{hparam}_avgToken/{model_name}"
            os.makedirs(emb_save_path, exist_ok=True)
            max_text_tokens = None

        elif token_type == "flatToken":
            print("Token type: Flat token embeddings")
            emb_save_path = f"./data/stim_features/vision-semantic/{hparam}_flatToken/{model_name}"
            os.makedirs(emb_save_path, exist_ok=True)
            max_text_tokens = max_text_token_count(frames_all, processor, annot_type)
            print(f"Max text tokens: {max_text_tokens}")
        # Iterate all videos
        first_key = next(iter(frames_all))
        first_value_list = frames_all[first_key]
        sample_img = first_value_list[0]   
        layer_num = get_layer_num(sample_img, model, model_name, processor, devices)
        for movname, frame_paths in frames_all.items():
            # Check if the embedding already exists
            exist_flag = False
            for layer_idx in range(layer_num):
                if os.path.exists(f"{emb_save_path}/layer{layer_idx}/{movname}.npy"):
                    print(f"Already exist: {movname} with {len(frame_paths)} frames")
                    exist_flag = True
            if exist_flag:
                continue
            frames_ds = downsample(frame_paths)
            captions = load_captions(movname, semantic_hparam)

            double_captions = []
            for item in captions:
                double_captions.append(item)
                double_captions.append(item)
            captions = double_captions
            print(movname)
            assert len(frames_ds)==len(captions), f"The number of frames does not match the number of annotations. {len(frames_ds)} != {len(captions)}"
            
            # Get embeddings
            print(f"Now processing: {movname} with {len(frame_paths)} frames and {len(captions)} captions")
            embs_all = convert_image_text_to_embedding(
                frames_ds, captions, processor, model, model_name, layer_num, devices, first_device,
                max_text_tokens, vis_token_num, token_type)
            for layer, embs in embs_all.items():
                print(f"{model_name} {layer}'s embedding shape: {embs.shape}")
                layer_save_path = f"{emb_save_path}/{layer}"
                os.makedirs(layer_save_path, exist_ok=True)
                np.save(f"{layer_save_path}/{movname}.npy", embs)
            

def convert_image_text_to_embedding(
    frames_ds, 
    captions, 
    processor, 
    model, 
    model_name,
    layer_num,
    devices,
    first_device,
    max_text_token,
    vis_token_num,
    token_type
    ):

    # Get number of layers
    embs_all = {f"layer{layer_idx+1}":[] for layer_idx in range(layer_num)}
    for frame_idx, (frame_path, caption) in enumerate(tqdm(zip(frames_ds, captions), total=len(frames_ds))):
        img = Image.open(frame_path)
        img_list = [img for _ in caption]
        with torch.no_grad():
            if model_name in ["LLaVA-v1.5-question-inputs"]:
                caption = [f"USER: <image>\n{cap} Is this description correct?\nASSISTANT:" for cap in caption]
            elif model_name in ["LLaVA-v1.5"]:
                caption = [f"USER: <image>\n{cap}\nASSISTANT:" for cap in caption]
            elif model_name in ["Kosmos-2"]:
                caption = [f"<grounding>{cap}:" for cap in caption]
                
            embs_annot = []
            # Average along the dimension of the annotators
            for (img, cap) in zip(img_list, caption):
                inputs = processor(
                    text=cap, images=img, return_tensors="pt"
                    )
                if len(devices) == 1:
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}
                embs = model(**inputs, output_hidden_states=True)
                embs = embs.hidden_states
                
                if model_name in ["BridgeTower"]:
                    embs_layer = []
                    embs = embs[2] # Multi-modal module
                    for emb in embs:
                        emb_sem = emb[0]
                        emb_vis = emb[1]
                        emb_sem = emb_sem.mean(axis=1)
                        if token_type == "avgToken":
                            emb_vis = emb_vis.mean(axis=1)
                        elif token_type == "flatToken":
                            emb_vis = emb_vis.flatten()
                        emb_sem = emb_sem.squeeze()
                        emb_vis = emb_vis.squeeze()
                        emb = torch.cat([emb_vis, emb_sem])
                        emb = emb.cpu().detach().numpy()
                        embs_layer.append(emb)
                    embs = np.array(embs_layer)
                    embs_annot.append(embs)
                
                else:
                    if model_name in ["LLaVA-v1.5", "LLaVA-v1.5-question-inputs"]:
                        image_token_mask = inputs["input_ids"] == model.config.image_token_index
                        image_token_pos = int(np.where(image_token_mask.cpu().numpy())[1])
                        emb = torch.cat(embs)
                        emb_vis = emb[:, image_token_pos:image_token_pos + vis_token_num, :]
                        emb_sem = torch.cat([emb[:, :image_token_pos, :], emb[:, image_token_pos + vis_token_num:, :]], dim=1)
                    elif model_name in ["GIT"]:
                        emb = torch.cat(embs)
                        vis_token_num = vis_token_num + 1
                        emb_vis = emb[:, :vis_token_num, :]
                        emb_sem = emb[:, vis_token_num:, :]
                    if token_type == "avgToken":
                        emb_vis = emb_vis.mean(axis=1)
                    elif token_type == "flatToken":
                        emb_vis = emb_vis.reshape(emb_vis.shape[0], -1)
                    emb_sem = emb_sem.mean(axis=1)
                    emb_sem = emb_sem.squeeze()
                    emb = torch.cat([emb_vis, emb_sem], dim=1)
                    emb = emb.cpu().detach().numpy()
                    embs_annot.append(emb)
                        
            embs = np.array(embs_annot)
            embs = embs.mean(axis=0)
            for layer_idx, emb in enumerate(embs):
                embs_all[f"layer{layer_idx+1}"].append(emb)
                    
    embs_all = {layer:np.array(embs) for layer, embs in embs_all.items()}
    # Averaging by 2 frames
    embs_all = {layer:embs.reshape(-1, 2, embs.shape[-1]).mean(axis=1) for layer, embs in embs_all.items()}
                    
    return embs_all


def get_layer_num(img_path, model, model_name, processor, devices):
    if len(devices) > 1:
        first_device = f"cuda:{devices[0]}"
    else:
        first_device = devices[0]
    img = Image.open(img_path)
    caption = "USER: <image>\nThese are two dogs lying on a pink couch. Is this description correct?\nASSISTANT:"
    if model_name in ["BridgeTower", "LLaVA-v1.5-question-inputs", "LLaVA-v1.5", "GIT"]:
        input = processor(text=caption, images=img, return_tensors="pt")
    elif model_name in ["Kosmos-2"]:
        caption = "<grounding>An image of"
        input = processor(text=caption, images=img, return_tensors="pt")
        
    if len(devices) == 1:
        input = {k: v.to(first_device) for k, v in input.items()}

    embs = model(**input, output_hidden_states=True)
    embs = embs.hidden_states
    if model_name in ["BridgeTower"]:
        layer_num = len(embs[2])
    else:
        layer_num = len(embs)
    print(f"Number of layers: {layer_num}")
    
    return layer_num


def max_text_token_count(frames_all, processor, annot_type):
    max_tokens = 0
    for movname, _ in frames_all.items():
        captions = load_captions(movname, annot_type)

        for text_list in captions:
            for text in text_list:
                try:
                    tokens = processor.tokenizer(text)
                except:
                    tokens = processor.tokenize(text)
                try:
                    max_tokens = max(max_tokens, len(tokens.input_ids))
                except:
                    max_tokens = max(max_tokens, len(tokens))
    
    return max_tokens