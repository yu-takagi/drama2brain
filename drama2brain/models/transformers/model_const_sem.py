from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2Model,
    LlamaTokenizer, LlamaModel,
    AutoTokenizer, OPTModel,
    AutoTokenizer, AutoModelForCausalLM,
)

model_dict_semantic = {
    "BERT":(BertTokenizer, BertModel,'bert-base-uncased'),
    "GPT2":(GPT2Tokenizer, GPT2Model,"gpt2-large"),
    "Llama2":(LlamaTokenizer, LlamaModel, "meta-llama/Llama-2-7b-hf"),
    "OPT":(AutoTokenizer, OPTModel, "facebook/opt-6.7b"),
    "Vicuna-v1.5":(AutoTokenizer, AutoModelForCausalLM, "lmsys/vicuna-7b-v1.5")
}