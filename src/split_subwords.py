"""
split_subwords_longest_match_first.py

Now extended to handle single strings OR a list of strings,
and pad subwords to the same length when processing a list.
"""

from typing import Union, List, Set


def _split_one_word(
    word: str,
    vocabulary: Set[str],
    skip_vocabulary: Set[str],
    subword_prefix: str,
    unknown_tag: str,
    pad_tag: str,
    scan_from_right: bool
) -> List[str]:
    """
    Split a single word into subwords, without padding.
    """

    # 1) If word is in skip vocabulary, return as-is:
    if word in skip_vocabulary:
        return [word]

    # 2) If the entire word is in vocabulary, no need to split.
    if word in vocabulary:
        return [word]

    # Reverse the characters if scanning from right
    chars = word[::-1] if scan_from_right else word

    start = 0
    max_len = len(chars)
    subwords = []

    while start < max_len:
        found_subword = None
        end = max_len

        while end > start:
            piece = chars[start:end]

            # piece is reversed in 'chars' if scanning from right,
            # so reverse it back to normal orientation to check in vocabulary
            if scan_from_right:
                piece = piece[::-1]

            # If it's not the first chunk, apply the prefix (e.g. '##')
            if start > 0:
                piece = subword_prefix + piece

            if piece in vocabulary:
                found_subword = piece
                break

            end -= 1

        if found_subword is None:
            # Cannot split => unknown
            return [unknown_tag]
        else:
            subwords.append(found_subword)
            start = end

    # If scanning from right, subwords were appended in reverse order,
    # so flip them to normal left-to-right reading order before returning.
    if scan_from_right:
        subwords.reverse()

    return subwords


def split_subwords_longest_match_first(
    words: Union[str, List[str], List[List[str]]],
    vocabulary: Set[str],
    skip_vocabulary: Set[str],
    subword_prefix: str,
    unknown_tag: str,
    pad_tag: str,
    scan_from_right: bool = False
) -> Union[List[str], List[List[str]]]:
    """
    Split input (scalar string, 1D list of strings, or 2D list of lists of strings)
    into subwords using a longest-match-first approach, flattening each row and
    padding it to the same length. Return shape based on input dimensionality:

      - scalar => 1D list of subwords
      - 1D => 1D list of subwords
      - 2D => 2D list of subwords
    """

    # Step 1: Detect the "dimensionality" of 'words'
    # --------------------------------------------
    if isinstance(words, str):
        # scalar (0D from the user perspective)
        # We'll internally treat as shape (1, 1)
        temp_2d = [[words]]
        input_dim = 0
    elif (
        isinstance(words, list) and len(words) > 0
        and all(isinstance(x, str) for x in words)
    ):
        # 1D list of strings
        # We'll treat as shape (1, N)
        temp_2d = [words]
        input_dim = 1
    elif (
        isinstance(words, list) and len(words) > 0
        and all(isinstance(row, list) and all(isinstance(x, str) for x in row)
                for row in words)
    ):
        # 2D list of lists of strings
        temp_2d = words
        input_dim = 2
    else:
        # Edge case: empty input or something else
        # We'll just return an empty list for simplicity
        return []

    # Step 2: For each row in this 2D structure,
    #         flatten all words in that row into a single list of subwords.
    # --------------------------------------------------------------------
    all_rows_subwords = []  # this will be List[List[str]]

    for row in temp_2d:
        row_subwords = []
        for w in row:
            splitted = _split_one_word(
                word=w,
                vocabulary=vocabulary,
                skip_vocabulary=skip_vocabulary,
                subword_prefix=subword_prefix,
                unknown_tag=unknown_tag,
                pad_tag=pad_tag,
                scan_from_right=scan_from_right
            )
            row_subwords.extend(splitted)
        all_rows_subwords.append(row_subwords)

    # Step 3: Find the maximum length of subwords among all rows
    # ----------------------------------------------------------
    max_len = max(len(r) for r in all_rows_subwords) if all_rows_subwords else 0

    # Step 4: Pad each row to max_len with pad_tag
    # --------------------------------------------
    for r_i, sub_list in enumerate(all_rows_subwords):
        diff = max_len - len(sub_list)
        if diff > 0:
            all_rows_subwords[r_i] = sub_list + [pad_tag] * diff

    # Step 5: Unbox shape based on original dimensionality
    # ----------------------------------------------------
    if input_dim == 0:
        # scalar => we had shape (1,1) => return a 1D list
        # i.e. all_rows_subwords[0] is the one row
        return all_rows_subwords[0]
    elif input_dim == 1:
        # 1D => we had shape (1,N) => return that single row as 1D
        return all_rows_subwords[0]
    else:
        # 2D => keep it 2D
        return all_rows_subwords


if __name__ == "__main__":
    vocab = {"cer", "##br", "##ec", "hello", "[UNK]", "[PAD]"}
    skip_vocab = {"[PAD]"}

    single_word = "cerbrec"
    result_single = split_subwords_longest_match_first(
        words=single_word,
        vocabulary=vocab,
        skip_vocabulary=skip_vocab,
        subword_prefix="##",
        unknown_tag="[UNK]",
        pad_tag="[PAD]",
        scan_from_right=False
    )
    print("Single word:", result_single)
    # e.g. ["cer", "##br", "##ec"]

    multiple_words = ["cerbrec", "hello", "hi"]
    result_list = split_subwords_longest_match_first(
        words=multiple_words,
        vocabulary=vocab,
        skip_vocabulary=skip_vocab,
        subword_prefix="##",
        unknown_tag="[UNK]",
        pad_tag="[PAD]",
        scan_from_right=False
    )
    print("List of words (padded):", result_list)
    # e.g. [
    #   ["cer", "##br", "##ec"],   # length=3
    #   ["hello", "[PAD]", "[PAD]"],  # length=3
    #   ["[UNK]", "[PAD]", "[PAD]"]   # length=3 (since "hi" is not in vocab => [UNK])
    # ]

    # 1) Scalar input
    single_word = "cerbrec"
    vocab = {"cer", "##br", "##ec", "hello", "[UNK]", "[PAD]"}
    skip_vocab = {"[PAD]"}

    res_scalar = split_subwords_longest_match_first(
        words=single_word,
        vocabulary=vocab,
        skip_vocabulary=skip_vocab,
        subword_prefix="##",
        unknown_tag="[UNK]",
        pad_tag="[PAD]",
        scan_from_right=False
    )
    print("Scalar input => 1D output:", res_scalar)
    # e.g. ["cer", "##br", "##ec"]

    # 2) 1D input
    words_1d = ["cerbrec", "hello", "hi"]
    res_1d = split_subwords_longest_match_first(
        words=words_1d,
        vocabulary=vocab,
        skip_vocabulary=skip_vocab,
        subword_prefix="##",
        unknown_tag="[UNK]",
        pad_tag="[PAD]",
        scan_from_right=False
    )
    print("1D input => 1D output:", res_1d)
    # e.g. ["cer", "##br", "##ec", "hello", "[UNK]", "[PAD]", "[PAD]"]
    #   => note it will flatten them: 'cerbrec' => ["cer","##br","##ec"],
    #      'hello' => ["hello"], 'hi' => ["[UNK]"], then all in one row
    #   => the row is padded to the length of the row with the most subwords (in this case 3 subwords from 'cerbrec')?
    #   => Actually we see we do a single flatten row. So the entire row length = 3 + 1 + 1 = 5 subwords
    #   => If there's a longer row for some reason, it would pad as needed.

    # 3) 2D input
    words_2d = [
        ["cerbrec", "hello"],
        ["hi", "[PAD]"]
    ]
    res_2d = split_subwords_longest_match_first(
        words=words_2d,
        vocabulary=vocab,
        skip_vocabulary=skip_vocab,
        subword_prefix="##",
        unknown_tag="[UNK]",
        pad_tag="[PAD]",
        scan_from_right=False
    )
    print("2D input => 2D output:", res_2d)
    # In row[0], we get ["cer","##br","##ec","hello"] => length 4
    # In row[1], we get ["[UNK]","[PAD]"] => length 2
    # So we pad row[1] to length 4 => ["[UNK]","[PAD]","[PAD]","[PAD]"]
    #
    # final:
    # [
    #   ["cer","##br","##ec","hello"],
    #   ["[UNK]","[PAD]","[PAD]","[PAD]"]
    # ]