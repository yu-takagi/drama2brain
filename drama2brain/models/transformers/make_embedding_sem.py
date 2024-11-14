from tqdm import tqdm
from drama2brain.models.transformers.model_const_sem import model_dict_semantic
from drama2brain.models.utils_models import compute_max_tokens
from drama2brain.utils import load_frames, load_captions
from drama2brain.data_const import STIM_DIR, FRAME_DIR
import os
import numpy as np
import torch


def embedding_maker_sem(annot_type, annotators_split_method, device, n_windows, avg_token) -> None:
    # Check the combination of input annot_type and annotators_split_method
    validate_combination(annot_type, annotators_split_method)

    model_names = model_dict_semantic
    for model_name, (tokenizer, model, model_path) in model_names.items():
        saved_model_path = os.path.join("./data/model_ckpts", model_path)
        if os.path.exists(saved_model_path):
            print(f"Already exist: Loading {model_name}'s weight from {saved_model_path}")
            model = model.from_pretrained(saved_model_path, torch_dtype=torch.float16, device_map=device)
            tokenizer = tokenizer.from_pretrained(saved_model_path)
        else:
            print(f"Now processing: Loading {model_name}'s weight from {model_path}")
            model = model.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
            tokenizer = tokenizer.from_pretrained(model_path)      
            
        source_directory = f'{FRAME_DIR}/frames'
        frames_all = load_frames(source_directory)

        # define max_length and max_tokens for model
        if model_name == 'GPT2':
            max_length = model.config.n_positions
        else:
            max_length = model.config.max_position_embeddings
        print(f'max_length = {max_length}')    
            
        print("Now computing max_tokens")
        if avg_token:
            emb_save_path = os.path.join(STIM_DIR, "semantic", annot_type+"_cw"+str(n_windows)+"_avgToken")
            max_tokens = max_length
        else:
            emb_save_path = os.path.join(STIM_DIR, "semantic", annot_type+"_cw"+str(n_windows)+"_flatToken")
            # set pad_token_id if there are not pad_token_id
            if "<pad>" not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"pad_token":"<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            max_tokens = compute_max_tokens(frames_all, "semantic_transformers", tokenizer, annot_type, n_windows, max_length)
        print(f'max_tokens = {max_tokens}')

        if annotators_split_method == "all":
            emb_save_path = os.path.join(emb_save_path+"_ALL", model_name)
        else:
            emb_save_path_A = os.path.join(emb_save_path+"_splitA", model_name) # object: 3 annotators, story: 1st annotators
            emb_save_path_B = os.path.join(emb_save_path+"_splitB", model_name) # object: 2 annotators, story: 2nd annotators

        model.eval()

        # Iterate all videos
        for movname, frame_paths in frames_all.items():
            if os.path.exists(f"{emb_save_path}/{movname}.npy"):
                print(f"Already exist: {annot_type}-{movname} with {len(frame_paths)} frames")
                continue
            print(f"Now processing: {annot_type}-{movname} with {len(frame_paths)} frames")

            captions_all = load_captions(movname, annot_type)
            
            # Make embedding from caption
            if annotators_split_method == "all":
                embs_all = convert_text_to_embedding(captions_all,
                                                     tokenizer, model,
                                                     annotators_split_method,
                                                     device,
                                                     n_windows,
                                                     max_tokens,
                                                     avg_token)
                print(f"Embedding's shape: (n_captions, n_layers, n_hidden_states) = {embs_all.shape}")
                
                for layer_idx in range(embs_all.shape[1]):
                    save_embs = embs_all[:,layer_idx,:].squeeze()
                    layer_save_path = f"{emb_save_path}/layer{layer_idx+1}"
                    os.makedirs(layer_save_path, exist_ok=True)
                    np.save(f"{layer_save_path}/{movname}.npy", save_embs)

                del embs_all, save_embs

            elif annotators_split_method == "split_3v2" or annotators_split_method == "split_1v1":
                embs_all_A, embs_all_B = convert_text_to_embedding(captions_all,
                                                                   tokenizer, model,
                                                                   annotators_split_method,
                                                                   device,
                                                                   n_windows,
                                                                   max_tokens,
                                                                   avg_token)
                print(f"Embedding's shape: (n_captions, n_layers, n_hidden_states) = {embs_all_A.shape}")
                    
                for embs_all, emb_save_path in zip([embs_all_A, embs_all_B], [emb_save_path_A, emb_save_path_B]):
                    for layer_idx in range(embs_all.shape[1]):
                        save_embs = embs_all[:,layer_idx,:].squeeze()
                        layer_save_path = f"{emb_save_path}/layer{layer_idx+1}"
                        os.makedirs(layer_save_path, exist_ok=True)
                        np.save(f"{layer_save_path}/{movname}.npy", save_embs)

                del embs_all_A, embs_all_B, embs_all, save_embs
    
    return None


def convert_text_to_embedding(captions_all,
                                tokenizer,
                                model, 
                                annotators_split_method,
                                device,
                                n_windows,
                                max_tokens,
                                avg_token):
    
    if annotators_split_method == "all":
        embs_all = []
    else:
        embs_all_A = []
        embs_all_B = []
    n_layers, padding, all_hidden_states = define_model_params(model, max_tokens, avg_token)

    for i in tqdm(range(len(captions_all))):
        embs_annotators = []
        # setting the range of time windows
        start_window = i-n_windows+1
        if start_window < 0:
            start_window = 0
        last_window = i+1
        captions = captions_all[start_window : last_window]

        for n in range(len(captions[0])):
            caption = np.array(captions)[:,n]
            unique_caption = np.unique(caption)
            unique_caption = ','.join([element for element in unique_caption if element != '..'])
            
            if unique_caption:
                prompts_ids = tokenizer(unique_caption, return_tensors="pt", padding=padding, max_length=max_tokens, truncation=True).input_ids.to(device)
                embs = model(prompts_ids, output_hidden_states=True)
                embs = embs.hidden_states
                embs_layers = []
                for emb in embs:
                    if avg_token:
                        emb = emb.mean(dim=1)
                    else:
                        emb = emb.flatten()
                    emb = emb.cpu().detach().numpy()
                    embs_layers.append(emb)
                embs_annotators.append(embs_layers)
            else:
                embs_annotators.append(None)

        if annotators_split_method == "all":

            embs_annotators = [embs_layer for embs_layer in embs_annotators if embs_layer is not None]

            if not embs_annotators:
                # setting zeros matrix as embeddings if all captions == '..'
                embs_annotators_mean = np.zeros((n_layers, all_hidden_states))
            else:
                # averaging each layer's embeddings between all annotators
                embs_annotators_mean = np.mean(embs_annotators, axis=0).squeeze()

            embs_all.append(embs_annotators_mean)
            
        elif annotators_split_method == "split_3v2": # For object annotators' QC
            
            embs_annotators_list = [embs_annotators[:3], embs_annotators[3:]]

            for j, embs_annotators in enumerate(embs_annotators_list):
                embs_annotators_split = [embs_layer for embs_layer in embs_annotators if embs_layer is not None]
                if not embs_annotators_split:
                    embs_annotators_split = np.zeros((n_layers, all_hidden_states))
                else:
                    embs_annotators_split = np.mean(embs_annotators_split, axis=0).squeeze()

                if j == 0:
                    embs_all_A.append(embs_annotators_split)
                else:
                    embs_all_B.append(embs_annotators_split)

        elif annotators_split_method == "split_1v1": # For story annotators' QC

            for j, embs_annotator in enumerate(embs_annotators):
                if embs_annotator == None:
                    embs_annotators_split = np.zeros((n_layers, all_hidden_states))
                else:
                    embs_annotators_split = np.array(embs_annotator).squeeze()
                
                if j == 0:
                    embs_all_A.append(embs_annotators_split)
                else:
                    embs_all_B.append(embs_annotators_split)     

    if annotators_split_method == "all":
        embs_all = np.array(embs_all)

        return embs_all # embs_all includes all layer's embeddings
    else:
        embs_all_A = np.array(embs_all_A)
        embs_all_B = np.array(embs_all_B)

        return embs_all_A, embs_all_B
    

def define_model_params(model, max_tokens, avg_token):
    n_hidden_states = model.config.hidden_size
    n_layers = model.config.num_hidden_layers+1

    if avg_token:
        print("Averaging tokens when making embeddings")
        padding = False
        all_hidden_states = n_hidden_states
    else:
        print("Flattening tokens when making embeddings")
        padding = "max_length"
        all_hidden_states = max_tokens * n_hidden_states
    
    return n_layers, padding, all_hidden_states


def validate_combination(annot_type, annotators_split_method):
    """
    Function to validate the combination of input annot_type and annotators_split_method
    """
    # Generalize conditions using a dictionary containing expected values
    expected_values = {
        "all": ["speechTranscription", "objectiveAnnot50chara", "story", "eventContent", "time_place"],
        "split_3v2": ["objectiveAnnot50chara"],
        "split_1v1": ["story"],
    }
    
    # Check conditions
    if annotators_split_method in expected_values:
        expected_b = expected_values[annotators_split_method]
        if isinstance(expected_b, list):
            assert annot_type in expected_b, "Review the combination of annot_type and annotators_split_method."
    else:
        raise ValueError("Review the combination of annot_type and annotators_split_method.")