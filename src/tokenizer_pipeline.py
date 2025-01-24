import re
import unicodedata
from typing import List

from src.tokenizer.byte_level import gpt2_byte_level_encode
from src.tokenizer.split_subwords import split_subwords_longest_match_first
from src.tokenizer.tokenizer_config import TokenizerConfig, extract_tokenizer_config, SubwordAlgorithm, \
    SpecialTokenStrategy


class TokenizerPipeline:
    def __init__(self, model_id):

        config_dict, vocab = extract_tokenizer_config(model_id)
        config = TokenizerConfig(**config_dict)

        # Subword method from config
        self.subword_method = config.tokenization_algorithm  # "bpe", "wordpiece", "sentencepiece", etc.

        # Determine special-token strategy from model_type (or from model_id)
        # hf_config = AutoConfig.from_pretrained(model_id)
        self.special_token_strategy = config.special_token_strategy

        self.config = config
        self.vocab = vocab
        self.vocab_set = set(vocab.keys())  # for quick membership checks


    def _normalize_text(self, text: str) -> str:
        """
        Decide how to normalize text, depending on subword_method and
        any known quirks (GPT-2 byte-level, T5, WordPiece lower-casing, etc.).
        """
        if self.subword_method == SubwordAlgorithm.bpe:
            return gpt2_byte_level_encode(text)

        elif self.subword_method == SubwordAlgorithm.sentencepiece:
            # Generic fallback
            text = unicodedata.normalize("NFKC", text)

            # Step 2) Usually SentencePiece replaces " " at word boundaries with '▁'
            # A minimal approach:
            #   - Insert '▁' at the start of the string if it starts with space
            #   - Then replace all sequences of spaces with "▁" + (the rest).
            #   - Typically you want to handle multiple spaces in a row, too.
            if text and not text.startswith(" "):
                text = " " + text

            # Make sure each "word boundary" space becomes '▁'
            # We'll do a naive approach:
            text = re.sub(r" +", lambda m: "▁" * len(m.group(0)), text)
            return text


        elif self.subword_method == SubwordAlgorithm.wordpiece:
            # BERT typically does lower-casing if do_lower_case = True.
            if self.config.do_lower_case:
                text = text.lower()
            # Possibly strip or unify whitespace
            return text.strip()

        # Fallback if unknown
        return text.strip()


    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Split the text at a higher level before subword splitting.
        For GPT-2 or BPE-based, you might split on whitespace or do minimal splitting.
        For WordPiece, you might do punctuation-splitting, etc.
        For sentencepiece you might skip this step or do minimal splitting.
        """
        if self.subword_method == SubwordAlgorithm.bpe:
            # GPT-2 or BPE approach => often just split on spaces.
            # (Because gpt2_byte_level_encode already handled the byte-level mapping.)
            return text.split()

        elif self.subword_method == SubwordAlgorithm.sentencepiece:
            # Typically you'd rely on the actual sp_model to do it internally,
            # but if you're in pure Python, you can do a minimal fallback:
            return text.split()

        elif self.subword_method == SubwordAlgorithm.wordpiece:
            # BERT-like punctuation splitting:
            # if you want to mimic the basic tokenizer’s punctuation logic:
            text = re.sub(r"([.,!?])", r" \1 ", text)
            return text.split()

        # Fallback
        return text.split()


    def _subword_tokenize(self, tokens: List[str]) -> List[str]:
        """ subword tokenize"""

        if self.config.tokenization_algorithm == SubwordAlgorithm.wordpiece:
            prefix = self.config.wordpiece_prefix or "##"
            skip_vocab = {self.config.special_tokens.pad_token}
            scan_from_right = False
            unknown_token = self.config.special_tokens.unk_token

        elif self.config.tokenization_algorithm == SubwordAlgorithm.bpe:
            prefix = self.config.continuing_subword_prefix or ""
            skip_vocab = {self.config.special_tokens.pad_token}
            scan_from_right = False
            unknown_token = self.config.special_tokens.unk_token

        elif self.config.tokenization_algorithm == SubwordAlgorithm.sentencepiece:
            prefix = ""  # T5 or LLaMA might not do prefix the same as WordPiece
            skip_vocab = set()
            scan_from_right = True
            unknown_token = self.config.special_tokens.unk_token

        else:
            raise ValueError(f"Unknown subword method: {self.config.tokenization_algorithm}")

        return split_subwords_longest_match_first(
            words=tokens,
            vocabulary=self.vocab_set,
            skip_vocabulary=skip_vocab,
            subword_prefix=prefix,
            unknown_tag=unknown_token,
            pad_tag=self.config.special_tokens.pad_token,
            scan_from_right=scan_from_right
        )

    def tokenize(self, text: str) -> List[int]:
        # 1) Normalize
        text = self._normalize_text(text)
        # 2) Basic tokenize
        tokens = self._basic_tokenize(text)
        # 3) Subword tokenize
        subwords = self._subword_tokenize(tokens)
        # 4) Possibly add special tokens
        subwords = self._add_special_tokens(subwords)
        # 5) Convert to IDs
        return [self._token_to_id(x) for x in subwords]

    def _add_special_tokens(self, tokens: List[str]) -> List[str]:
        """
        Decide how (and if) to add special tokens based on self.special_token_strategy.
        """
        if self.special_token_strategy == SpecialTokenStrategy.bert_style:
            # WordPiece => add [CLS] ... [SEP], if in vocab
            has_cls = self.config.special_tokens.cls_token in self.vocab
            has_sep = self.config.special_tokens.sep_token in self.vocab
            if has_cls and has_sep:
                return [self.config.special_tokens.cls_token] + tokens + [self.config.special_tokens.sep_token]

        elif self.special_token_strategy == SpecialTokenStrategy.bart_style:
            # BART/RoBERTa => add <s> ... </s> if they exist
            has_bos = self.config.special_tokens.bos_token in self.vocab
            has_eos = self.config.special_tokens.eos_token in self.vocab
            if has_bos and has_eos:
                return [self.config.special_tokens.bos_token] + tokens + [self.config.special_tokens.eos_token]

        elif self.special_token_strategy == SpecialTokenStrategy.gpt2_style:
            # GPT-2 => typically no special tokens for single-sentence
            return tokens

        elif self.special_token_strategy == SpecialTokenStrategy.llama_style:
            # LLaMA => also typically no special tokens for single-sentence
            return tokens

        elif self.special_token_strategy == SpecialTokenStrategy.t5_style:
            # T5 => no BOS, but does use </s> as EOS
            has_eos = self.config.special_tokens.eos_token in self.vocab
            if has_eos:
                return tokens + [self.config.special_tokens.eos_token]

        # fallback or "none" => no special tokens
        return tokens

    def _token_to_id(self, token: str) -> int:
        """
        Convert string token to an integer ID. If not found, use [UNK] token ID (or fallback 0).
        """
        unk_id = self.vocab.get(self.config.special_tokens.unk_token, 0)
        return self.vocab.get(token, unk_id)