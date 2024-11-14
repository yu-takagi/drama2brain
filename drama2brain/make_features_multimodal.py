"""
python3 -m drama2brain.make_features_multimodal \
    --modality vision-semantic \
    --modality_hparam default-story \
    --model_source transformers \
    --devices 0 \
    --n_windows 1 \
    --token_type avgToken
    
annot_type = ["objectiveAnnot50chara", "story", "speechTranscription", "eventContent", "timePlace"]

"""
import argparse
from drama2brain.models.transformers.download_model import transformers_loader
from drama2brain.models.transformers.make_embedding_vis_sem  import embedding_maker_vis_sem
from drama2brain.models.transformers.make_embedding_vis  import embedding_maker_vis
from drama2brain.models.transformers.make_embedding_audio  import embedding_maker_audio

def main(args):
    if args.model_source == "transformers":
        if args.modality == "vision-semantic":
            print("Now downloading transformers models...")
            transformers_loader(args.modality)
            for modality_hparam in args.modality_hparam:
                print(f"Now making {modality_hparam} embeddings from transformers models...")
                embedding_maker_vis_sem(args.devices, modality_hparam, args.token_type)
        elif args.modality == "vision":
            print("Now downloading transformers models...")
            transformers_loader(args.modality)
            print(f"Now making {args.modality} embeddings from transformers models...")
            embedding_maker_vis(args.devices, args.token_type)
        elif args.modality == "audio":
            print("Now downloading transformers models...")
            transformers_loader(args.modality)
            print(f"Now making {args.modality} embeddings from transformers models...")
            embedding_maker_audio(args.devices, args.token_type)
    
    else:
        raise NotImplementedError()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embeddings from LLMs to make multi-level semantic features"
    )
    parser.add_argument("--modality", type=str, default="semantic")
    parser.add_argument("--modality_hparam", nargs="*", type=str, required=True)
    parser.add_argument("--model_source", choices=["transformers", "gensim"], default="transformers")
    parser.add_argument("--devices", nargs="*", type=int, default=0)
    parser.add_argument("--n_windows", type=int, default=20)
    parser.add_argument("--token_type", choices=["avgToken", "flatToken"])

    args = parser.parse_args()
    main(args)