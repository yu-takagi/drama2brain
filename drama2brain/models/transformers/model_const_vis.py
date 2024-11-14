from transformers import (
    AutoFeatureExtractor, DeiTForImageClassificationWithTeacher,
    AutoImageProcessor, ResNetForImageClassification,
    AutoProcessor, LlavaForConditionalGeneration
)

model_dict_vision = {
    "DeiT": (AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, "facebook/deit-base-distilled-patch16-224"),
    "ResNet": (AutoImageProcessor, ResNetForImageClassification, "microsoft/resnet-50"),
    "CLIP": (AutoProcessor, LlavaForConditionalGeneration, "llava-hf/llava-1.5-7b-hf"),
}