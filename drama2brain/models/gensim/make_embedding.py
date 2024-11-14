"""
### What to do before you run this code

### For glove
(See more info: https://nlp.stanford.edu/projects/glove/)
1. wget https://nlp.stanford.edu/data/glove.840B.300d.zip
2. unzip glove.840B.300d.zip -d . && rm glove.840B.300d.zip

### For word2vec (You need a google account)
(See more info: https://code.google.com/archive/p/word2vec/)
1. Download GoogleNews-vectors-negative300.bin.gz 
   from https://code.google.com/archive/p/word2vec/
2. gzip -d GoogleNews-vectors-negative300.bin.gz
"""

from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from drama2brain.models.utils_models import compute_max_tokens
from drama2brain.utils import load_frames, load_captions
from drama2brain.data_const import STIM_DIR, FRAME_DIR
import os
import numpy as np

def embedding_maker(annot_type, n_windows, avg_token):
    model_names = {"GloVe": GloVe, "Word2Vec": Word2Vec}
    for model_name, model in model_names.items():
        print(model_name)
        weights_dir = "./data/gensim_models"
        model = model(weights_dir)

        source_directory = f'{FRAME_DIR}/frames'
        frames_all = load_frames(source_directory)

        if avg_token:
            emb_save_path = os.path.join(STIM_DIR ,"semantic", annot_type+"_cw"+str(n_windows)+"_avgToken_ALL", model_name, "layerNone")
            max_tokens=None
        else:
            emb_save_path = os.path.join(STIM_DIR ,"semantic", annot_type+"_cw"+str(n_windows)+"_flatToken_ALL", model_name, "layerNone")
            max_tokens = compute_max_tokens(frames_all, "semantic_gensim", model, annot_type, n_windows)
        print(f'max_tokens = {max_tokens}')

        # Iterate all videos
        for movname, frame_paths in frames_all.items():
            if os.path.exists(f"{emb_save_path}/{movname}.npy"):
                print(f"Already exist: {annot_type}-{movname} with {len(frame_paths)} frames")
                continue
            print(f"Now processing: {annot_type}-{movname} with {len(frame_paths)} frames")

            captions_all = load_captions(movname, annot_type)
            # Make embedding from caption
            embs_all = convert_text_to_embedding(captions_all, model, n_windows, max_tokens, avg_token)
            print(f"Embedding's shape: (n_captions, n_vectors) = {embs_all.shape}")

            #save embeddings
            os.makedirs(emb_save_path, exist_ok=True)
            np.save(f"{emb_save_path}/{movname}.npy", embs_all)

def convert_text_to_embedding(captions_all, model, n_windows, max_tokens, avg_token, n_vectors = 300):
    embs_all = []
    model_key2index, all_hidden_states = define_model_params(model, max_tokens, avg_token, n_vectors)
    
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
                words = unique_caption.lower().split()
                vectors = []
                for word in words:
                    if word in model_key2index:
                        vectors.append(model.get_vector(word))
                vectors = np.array(vectors)
                if avg_token:
                    vectors = np.mean(vectors, axis=0)
                else:
                    vectors = vectors.flatten()

                try:
                    if vectors.shape[0] % n_vectors == 0:
                        vectors.resize((1,all_hidden_states))
                        embs_annotators.append(vectors)
                except:
                    print('Ignoring because the dictionary has no word in captions')
                    pass

        if embs_annotators == []:
            # setting zeros matrix as embeddings if all captions == '..'
            embs_annotators_mean = np.zeros((all_hidden_states,))
        else:
            # averaging embeddings between all annotators
            embs_annotators_mean = np.mean(embs_annotators, axis=0).squeeze()

        embs_all.append(embs_annotators_mean)
    embs_all = np.array(embs_all)
    return embs_all


def GloVe(weights_dir, weights="glove.840B.300d.txt"):
    weights_file = os.path.join(weights_dir, "glove", weights)
    word2vec_weights_file = weights_file + '.word2vec'
    if not os.path.isfile(word2vec_weights_file):
        _ = glove2word2vec(weights_file, word2vec_weights_file)
    model = KeyedVectors.load_word2vec_format(word2vec_weights_file, binary=False)
    return model

def Word2Vec(weights_dir, weights="GoogleNews-vectors-negative300.bin"):
    weights_file = os.path.join(weights_dir, "word2vec", weights)
    model = KeyedVectors.load_word2vec_format(weights_file, binary=True)
    return model

def define_model_params(model, max_tokens, avg_token, n_vectors):
    model_key2index = model.key_to_index

    if avg_token:
        print("Averaging words when making embeddings")
        all_hidden_states = n_vectors
    else:
        print("Flattening words when making embeddings")
        all_hidden_states = max_tokens * n_vectors
    
    return model_key2index, all_hidden_states