from transformers import (
    AutoFeatureExtractor, ASTForAudioClassification,
    AutoProcessor, Wav2Vec2ForCTC,
)

model_dict_audio = {
    "AST": (AutoFeatureExtractor, ASTForAudioClassification, "MIT/ast-finetuned-audioset-10-10-0.4593"),
    "MMS": (AutoProcessor, Wav2Vec2ForCTC, "facebook/mms-1b-all"),
}