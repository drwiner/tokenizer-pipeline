from typing import Any, Dict, Tuple
from typing import Optional, List, Union, Literal

from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoConfig

from enum import Enum

class SpecialTokenStrategy(str, Enum):
    bert_style = "bert-style"
    bart_style = "bart-style"
    gpt2_style = "gpt2-style"
    llama_style = "llama-style"
    t5_style = "t5-style"
    none = "none"


class SubwordAlgorithm(str, Enum):
    wordpiece = "wordpiece"
    bpe = "bpe"
    sentencepiece = "sentencepiece"
    unknown = "unknown"


def get_special_token_strategy(model_type: str) -> SpecialTokenStrategy:
    """
    Return a string or enum describing how we should add special tokens
    based on model_type from AutoConfig.
    """

    if model_type in {"bert", "distilbert"}:
        return SpecialTokenStrategy.bert_style   # WordPiece style => add [CLS], [SEP]
    elif model_type in {"roberta", "bart", "marian", "mbart"}:
        return SpecialTokenStrategy.bart_style   # bos_token=<s>, eos_token=</s>
    elif model_type in {"gpt2", "bloom"}:
        return SpecialTokenStrategy.gpt2_style   # no special tokens by default
    elif model_type in {"llama"}:
        return SpecialTokenStrategy.llama_style  # typically no special tokens for single text,
                        # but user might want <s>, </s>
    elif model_type in {"t5", "mt5"}:
        return SpecialTokenStrategy.t5_style  # Uses </s> as EOS token, no BOS token by default

    # fallback
    return SpecialTokenStrategy.none


def guess_subword_algorithm(hf_tokenizer) -> SubwordAlgorithm:
    """
    Return 'wordpiece', 'bpe', 'sentencepiece', or 'unknown'
    based on Hugging Face tokenizer attributes.
    """
    if hasattr(hf_tokenizer, "sp_model"):
        return SubwordAlgorithm.sentencepiece # e.g. LLaMA, T5, Marian, etc.
    elif hasattr(hf_tokenizer, "encoder"):
        return SubwordAlgorithm.bpe           # e.g. GPT-2, RoBERTa, Bart, etc.
    elif hasattr(hf_tokenizer, "vocab"):
        return SubwordAlgorithm.wordpiece     # e.g. BERT, DistilBERT
    # elif hasattr(hf_tokenizer, "encoder"):
    #     return "bpe"           # e.g. GPT-2, RoBERTa, Bart, etc.
    else:
        return SubwordAlgorithm.unknown


def extract_tokenizer_config(model_id: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Build a config + vocab dict from a HF AutoTokenizer, in pure Python.
    """
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    # hf_config = AutoConfig.from_pretrained(model_id)

    # Step 1) Identify the model_type (roberta, bert, etc.) from hf_config
    # model_type = hf_config.model_type.lower()  # e.g. "roberta", "bert"

    # Load the HF config
    hf_config = AutoConfig.from_pretrained(model_id)

    # e.g. hf_config.model_type could be: "bert", "gpt2", "bloom", "roberta", "bart", ...
    model_type = hf_config.model_type.lower()

    # Step 2) Identify the subword algorithm from the class name or from logic
    tokenizer_class_name = hf_tokenizer.__class__.__name__
    special_token_strategy = get_special_token_strategy(model_type)
    tokenization_algo = guess_subword_algorithm(hf_tokenizer)

    # Step 3) Build the config dict
    config_dict = {
        "tokenizer_class": tokenizer_class_name, # Class Name
        "tokenization_algorithm": tokenization_algo, # Subword method
        "special_token_strategy": special_token_strategy, # Special token style.
        "model_max_length": hf_tokenizer.model_max_length,
        "padding_side": hf_tokenizer.padding_side,
        "truncation_side": getattr(hf_tokenizer, "truncation_side", "right"),
        "do_lower_case": getattr(hf_tokenizer, "do_lower_case", False),
        "special_tokens": {
            "bos_token": str(hf_tokenizer.bos_token),
            "eos_token": str(hf_tokenizer.eos_token),
            "unk_token": str(hf_tokenizer.unk_token),
            "sep_token": str(hf_tokenizer.sep_token),
            "pad_token": str(hf_tokenizer.pad_token),
            "cls_token": str(hf_tokenizer.cls_token),
            "mask_token": str(hf_tokenizer.mask_token),
            "additional_special_tokens": hf_tokenizer.additional_special_tokens or []
        }
    }

    vocab_dict: Dict[str, int] = {}

    # Step 4) Retrieve the underlying subword->ID mapping
    if hasattr(hf_tokenizer, "vocab"):
        vocab_dict = dict(hf_tokenizer.vocab)  # typical for WordPiece or BERT
    elif hasattr(hf_tokenizer, "encoder"):
        vocab_dict = dict(hf_tokenizer.encoder)  # GPT-2 style
    elif hasattr(hf_tokenizer, "sp_model"):
        spm_size = hf_tokenizer.sp_model.GetPieceSize()
        for i in range(spm_size):
            piece = hf_tokenizer.sp_model.IdToPiece(i)
            vocab_dict[piece] = i

    # Add the final vocab_size
    config_dict["vocab_size"] = len(vocab_dict)

    # Possibly store model_type in config_dict so you can look it up later
    config_dict["model_type"] = model_type

    return config_dict, vocab_dict


class SpecialTokens(BaseModel):
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    unk_token: str = "[UNK]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"
    mask_token: str = "[MASK]"
    additional_special_tokens: List[str] = Field(default_factory=list)

    def all_tokens_set(self) -> set:
        """
        Return a set of all special token string values, for quick membership checks.
        """
        return {
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token,
            *self.additional_special_tokens
        }

class TokenizerConfig(BaseModel):
    tokenizer_class: str  # e.g. "GPT2TokenizerFast"
    tokenization_algorithm: SubwordAlgorithm = SubwordAlgorithm.wordpiece
    special_token_strategy: SpecialTokenStrategy = SpecialTokenStrategy.none

    model_max_length: int
    vocab_size: int
    padding_side: Literal["right", "left"] = "right"
    truncation_side: Literal["right", "left"] = "right"

    # Preprocessing
    do_lower_case: bool = True
    clean_up_tokenization_spaces: bool = True
    strip_accents: Optional[bool] = None
    handle_chinese_chars: bool = True

    # Special tokens
    special_tokens: SpecialTokens = SpecialTokens()

    # Tokenizer-specific parameters
    wordpiece_prefix: Optional[str] = "##"  # for WordPiece
    continuing_subword_prefix: Optional[str] = ""  # for BPE
    end_of_word_suffix: Optional[str] = ""  # not used here, but you can store it
    byte_fallback: bool = False
    unicode_normalizer: Optional[str] = None

    # Advanced configuration
    add_prefix_space: bool = False
    trim_offsets: bool = True
    split_special_tokens: bool = False

    # Model-specific settings
    pad_token_type_id: int = 0
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Additional configurations
    extra_config: Dict[str, Union[str, int, bool, List, Dict]] = Field(default_factory=dict)



if __name__ == "__main__":
    # Usage
    _model_id = "bert-base-uncased"
    _config_dict, _ = extract_tokenizer_config(_model_id)
    _tokenizer_config = TokenizerConfig(**_config_dict)
    print(_tokenizer_config)