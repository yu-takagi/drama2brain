"""
python3 -m drama2brain.make_features_multiLevel --device 1 --annot_type story --annotators_split_method all --avg_token
annot_type = ["objectiveAnnot50chara", "story", "speechTranscription", "eventContent", "timePlace"]
"""
import argparse
from drama2brain.models.transformers.download_model import transformers_loader
from drama2brain.models.transformers.make_embedding_sem import embedding_maker_sem
from drama2brain.models.gensim.make_embedding import embedding_maker


def main(args):
    if args.device >= 0:
        device = f"cuda:{args.device}"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    if args.model_sourse == "transformers":
        print("Now downloading transformers models...")
        transformers_loader(args.modality)
        
        print("Now making embeddings from transformers models...")
        embedding_maker_sem(args.annot_type, args.annotators_split_method, device, args.n_windows, args.avg_token)

    elif args.model_sourse == "gensim":
        print("Now making embeddings from gensim models...")
        embedding_maker(args.annot_type, args.n_windows, args.avg_token)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embeddings from LLMs to make multi-level semantic features"
    )
    parser.add_argument("--modality", type=str, default="semantic")
    parser.add_argument("--annot_type", type=str, required=True)
    parser.add_argument("--annotators_split_method", choices=["all", "split_3v2", "split_1v1"], default="all")
    parser.add_argument("--model_sourse", choices=["transformers", "gensim"], default="transformers")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_windows", type=int, default=1)
    parser.add_argument("--avg_token", action='store_true')

    args = parser.parse_args()
    main(args)