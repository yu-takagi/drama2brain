from transformers import (
    AutoProcessor, AutoModel,
    BridgeTowerProcessor, BridgeTowerModel,
    AutoProcessor, LlavaForConditionalGeneration,
)

model_dict_vis_sem = {
    "GIT": (AutoProcessor, AutoModel, "microsoft/git-base"),
    "BridgeTower": (BridgeTowerProcessor, BridgeTowerModel, "BridgeTower/bridgetower-base"),
    "LLaVA-v1.5": (AutoProcessor, LlavaForConditionalGeneration, "llava-hf/llava-1.5-7b-hf"),
}