import numpy as np
import tqdm
from drama2brain.utils import load_captions

def compute_max_tokens(frames_all, modality, processor, annot_type, n_windows, max_length=None):
    max_tokens = 0
    if modality=="semantic_gensim":
        model_key2index = processor.key_to_index
        print("Now computing max tokens for Gensim model")
    elif modality=="semantic_transformers":
        print("Now computing max tokens for LLM")
        pass

    for movname, _ in frames_all.items():
        print(f"Now processing: {annot_type}-{movname}")
        captions_all = load_captions(movname, annot_type)
        for i in tqdm(range(len(captions_all))):
            start_window = i-n_windows+1
            if start_window < 0:
                start_window = 0
            last_window = i+1
            captions = captions_all[start_window: last_window]
            for n in range(len(captions[0])):
                caption = np.array(captions)[:,n]
                unique_caption = np.unique(caption)
                unique_caption = ','.join([element for element in unique_caption if element != '..'])
                if unique_caption:
                    if modality=="semantic_gensim":
                        words = unique_caption.lower().split()
                        tokens = []
                        for word in words:
                            if word in model_key2index:
                                tokens.append(processor.get_vector(word))
                    elif modality=="semantic_transformers":
                        tokens = processor(unique_caption, max_length=max_length, truncation=True).input_ids
                    max_tokens = max(max_tokens, len(tokens))
    return max_tokens