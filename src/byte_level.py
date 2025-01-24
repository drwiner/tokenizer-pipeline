""" Byte-level tokenization for GPT-2. """

def bytes_to_unicode():
    """
    Build a mapping from bytes to a range of unicode characters (u0100..u017F, etc.).
    Hugging Face does something like this to avoid conflicts with real text.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(x) for x in cs]
    return dict(zip(bs, cs))

byte2unicode = bytes_to_unicode()
unicode2byte = {v: k for k, v in byte2unicode.items()}

def gpt2_byte_level_encode(text: str) -> str:
    """
    Convert a Python string into GPT-2's "byte-level" string (basically each char
    is a mapped-from original byte).
    """
    # Turn input text => raw bytes
    text_bytes = text.encode('utf-8')
    # Map each byte b to a single "extended" unicode char
    return "".join(byte2unicode[b] for b in text_bytes)

def basic_gpt2_tokenize(text: str) -> str:
    """
    GPT-2 also merges spaces into a special 'Ġ' or not. Actually, GPT-2 doesn't
    do the same punctuation split as BERT. So we usually just keep it as-is,
    then do subword.
    """
    # Minimal: just do the byte-level encode
    return gpt2_byte_level_encode(text)