from tqdm import tqdm
from drama2brain.models.transformers.model_const_audio import model_dict_audio
import os
import numpy as np
from transformers import AutoProcessor
import torch
from datasets import Dataset, Audio
import glob
import gc
import torch.nn as nn
from drama2brain.data_const import AUDIO_DIR


def embedding_maker_audio(devices, input_seconds=1):
    model_names = model_dict_audio
    if len(devices) > 1:
        first_device = f"cuda:{devices[0]}"
    else:
        first_device = devices[0]
    
    for model_name, (processor, model, model_path) in model_names.items():
        print(model_name)

        source_directory = f"./{AUDIO_DIR}/audio"
        saved_model_path = f"./data/model_ckpts/audio/{model_path}"
        model = model.from_pretrained(saved_model_path)
        processor = processor.from_pretrained(saved_model_path)
        
        # Decoderがあるモデルのために、start_token_idを事前に取得
        try:
            decoder_start_token_id = model.config.decoder_start_token_id
        except:
            pass

        proc_audio_all = preprocess_audio(
            source_directory, model_name, processor, input_seconds
            )
        
        # Iterate all audios
        for audio_name, (proc_audio, audio_seconds) in proc_audio_all.items():
            if os.path.exists(f"{emb_save_path}/{audio_name}.npy"):
                print(f"Already exist: {audio_name} with {len(proc_audio)} frames")
                continue
            print(f"Now processing: {audio_name} with {len(proc_audio)} seconds")
            
            if len(devices) > 1:
                model = nn.DataParallel(model, device_ids=devices)
            model = model.to(first_device)
            
            print(f"Now extracting {audio_name}'s features...")
            embs_all = convert_audio_to_embedding(
                proc_audio, model, model_name, devices, decoder_start_token_id
                )
            
            if audio_name == "GIS2_002":
                audio_seconds = 852
                
            emb_save_path = os.path.join(f"./data/stim_features/audio/{input_seconds}s/", model_name)
            for layer, embs in embs_all.items():
                print(f"{model_name} {layer}'s embedding shape: {embs.shape}")
                layer_save_path = f"{emb_save_path}/{layer}"
                os.makedirs(layer_save_path, exist_ok=True)
                embs = embs[:audio_seconds]
                np.save(f"{layer_save_path}/{audio_name}.npy", embs)
                
            
def convert_audio_to_embedding(
    proc_audio,
    model,
    model_name,
    devices,
    decoder_start_token_id=0
    ):
    
    embs_all = {}
    # dec_embs_all = {}
    for segment_idx, inputs in enumerate(tqdm(proc_audio)):
        with torch.no_grad():
            if len(devices)==1:
                first_device = f"cuda:{devices[0]}"
                inputs = inputs.to(first_device)
                
            if model_name in ["Speech2Text", "Whisper-v3"]:
                decoder_input_ids = torch.tensor([[1, 1]]) * decoder_start_token_id
                decoder_input_ids = decoder_input_ids.to(first_device)
                embs = model(**inputs, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
            elif model_name in ["Encodec"]:
                embs = model(inputs["input_values"], inputs["padding_mask"], output_hidden_states=True)
            else:
                embs = model(**inputs, output_hidden_states=True)
                
            try:
                embs = embs.hidden_states
            except:
                embs = embs.encoder_hidden_states
            
            if segment_idx==0:
                embs_all = {f"layer{layer_idx+1}":[] for layer_idx in range(len(embs))}
                
            for layer_idx, emb in enumerate(embs):
                emb = emb.flatten()
                emb = emb.cpu().detach().numpy()
                embs_all[f"layer{layer_idx+1}"].append(emb)

    embs_all = {layer:np.array(embs) for layer, embs in embs_all.items()}

    del inputs, embs
    gc.collect()
    torch.cuda.empty_cache()
    
    return embs_all

def load_audio(audio_topdir):
    audio_dirs = [dir_path for dir_path in glob.glob(os.path.join(audio_topdir, "*"))]
    audio_dirs = sorted(audio_dirs)
    return audio_dirs

def segement_audio(dataset, input_seconds):
    audio = dataset["audio"][0]["array"]
    audio_sample_rate = dataset["audio"][0]["sampling_rate"]
    segment_length = input_seconds * audio_sample_rate
    audio_seconds = int(round(len(audio) / audio_sample_rate))
    if len(audio) != audio_seconds * audio_sample_rate:
        last_audio = audio[-1]
        padding_length = audio_seconds * audio_sample_rate - len(audio)
        audio = np.concatenate((audio, np.array([last_audio]*padding_length)))
    segments = []
    for end_seconds in range(1, audio_seconds+1, 1):
        end = end_seconds * audio_sample_rate
        start = max(0, end - segment_length)
        segment = audio[start:end]
        
        # edge processing
        if len(segment) != segment_length:
            repeat_segment = audio[0: audio_sample_rate]
            while len(segment) != segment_length:
                segment = np.concatenate((repeat_segment, segment))

        segments.append(segment)
        
    return segments, audio_seconds

def preprocess_audio(audio_dir: str, model_name:str, processor: AutoProcessor, input_seconds):
    audio_paths = load_audio(audio_dir)
    try:
        model_sample_rate = processor.sampling_rate
    except:
        model_sample_rate = processor.feature_extractor.sampling_rate

    audio_dict = {}
    for i, audio_path in enumerate(audio_paths):
        audio_name = os.path.basename(audio_path)
        audio_name = audio_name.replace(".wav", "")
        dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=model_sample_rate))
        
        # Split the audio data into segments according to the input duration required by the model.
        segments, audio_seconds = segement_audio(dataset, input_seconds)
        print(f"{audio_name}'s duraion: {audio_seconds} seconds")
        
        # Preprocess the segmented audio data.
        input_features_list = []
        for segment in segments:
            inputs = processor(segment, return_tensors="pt", sampling_rate=model_sample_rate)
            input_features_list.append(inputs)
        audio_dict[audio_name] = input_features_list, audio_seconds

    return audio_dict
