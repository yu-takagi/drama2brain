import os
import glob

def transformers_loader(modality):
    if modality == 'vision':
        from drama2brain.models.transformers.model_const_vis import model_dict_vision
        model_names = model_dict_vision
    elif modality == 'semantic':
        from drama2brain.models.transformers.model_const_sem import model_dict_semantic
        model_names = model_dict_semantic
    elif modality == "audio":
        from drama2brain.models.transformers.model_const_audio import model_dict_audio
        model_names = model_dict_audio
    elif modality == "vision-semantic":
        from drama2brain.models.transformers.model_const_vis_sem import model_dict_vis_sem
        model_names = model_dict_vis_sem

    for model_name, model_config in model_names.items():
        print(model_name)
    
        processor, model, model_path = model_config
        ckpt_save_path = os.path.join(f"./data/model_ckpts/{modality}", model_path)
        os.makedirs(ckpt_save_path, exist_ok=True)
        if glob.glob(os.path.join(ckpt_save_path, "*.safetensors")):
            print(f"{model_name}'s checkpoint already exists.")
            continue

        model = model.from_pretrained(model_path)
        processor = processor.from_pretrained(model_path)
        
        model.save_pretrained(ckpt_save_path, safe_serialization=True)
        processor.save_pretrained(ckpt_save_path, safe_serialization=True)