import argparse
import json
from pathlib import Path

"""
A very flexible tokenizer script that generates vocab.bin from tokenizer.json and auto-detects byte-level tokens
with flags for forcing or disbaling byte-level tokenization.
"""

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_TOKENIZER_FILENAME = "tokenizer.json"
DEFAULT_OUTPUT = "vocab.bin"


def _download_tokenizer_json(model_name: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download tokenizer.json. "
            "Install it or use --local-tokenizer-json."
        ) from exc
    path = hf_hub_download(model_name, filename=filename)
    return Path(path)


def _load_tokenizer_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_added_tokens(data: dict) -> list[dict]:
    added = data.get("added_tokens", [])
    return added if isinstance(added, list) else []


def _set_token(vocab: list, idx: int, token: str, source: str) -> None:
    existing = vocab[idx]
    if existing is None:
        vocab[idx] = token
        return
    if existing != token:
        raise ValueError(
            f"token id {idx} conflict between {source} ('{token}') and existing ('{existing}')"
        )


def _build_vocab_list(data: dict) -> list[str]:
    model = data.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json missing 'model' object")

    vocab = model.get("vocab")
    added_tokens = _collect_added_tokens(data)
    added_ids = [t["id"] for t in added_tokens if isinstance(t, dict) and "id" in t]
    max_added_id = max(added_ids) if added_ids else -1

    vocab_ids = list(vocab.values())
    if not vocab_ids:
        raise ValueError("tokenizer.json contains empty vocab")
    max_id = max(max(vocab_ids), max_added_id)
    vocab_list: list[str | None] = [None] * (max_id + 1)
    for token, id_ in vocab.items():
        if not isinstance(id_, int):
            raise ValueError("token id must be int")
        if id_ < 0:
            raise ValueError(f"token id must be non-negative (got {id_})")
        _set_token(vocab_list, id_, token, "model.vocab")

    # print(f"{vocab_list=}")
    for entry in added_tokens:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        token = entry.get("content")
        if idx is None or token is None:
            continue
        if not isinstance(idx, int):
            raise ValueError("added token id must be int")
        if idx < 0:
            raise ValueError(f"added token id must be non-negative (got {idx})")
        if idx >= len(vocab_list):
            print("vocab", [None] * (idx + 1 - len(vocab_list)))
            vocab_list.extend([None] * (idx + 1 - len(vocab_list)))
        _set_token(vocab_list, idx, token, "added_tokens")

    missing = [i for i, token in enumerate(vocab_list) if token is None]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        raise ValueError(
            f"tokenizer vocab has missing ids (count={len(missing)}), first: {preview}"
        )

    return [token for token in vocab_list if token is not None]


def _bytes_to_unicode() -> dict[int, str]:
    """
    A look-up table that maps byte value to a printable unicode character
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    b2u = dict(zip(bs, [chr(n) for n in cs]))
    return b2u


def _byte_decoder() -> dict[str, int]:
    """
    Maps printable unicode character to a byte value to translate placeholder value to actual
    byte value
    """
    decode_bytes = {v: k for k, v in _bytes_to_unicode().items()}
    return decode_bytes


def _has_byte_level_component(data: object) -> bool:
    if isinstance(data, dict):
        if data.get("type") == "ByteLevel":
            return True
        if data.get("byte_fallback") is True:
            return True
        return any(_has_byte_level_component(v) for v in data.values())
    if isinstance(data, list):
        return any(_has_byte_level_component(v) for v in data)
    return False


def _vocab_looks_byte_level(vocab: list[str]) -> bool:
    for token in vocab:
        for ch in token:
            if ord(ch) > 255:
                return True
    return False


def _encode_token_to_bytes(token: str, decoder: dict[str, int] | None) -> bytes:
    """
    Get the byte-level representation of a token
    Uses decoder
    """
    if decoder is None:
        return token.encode("utf-8")
    buf = bytearray()
    for ch in token:
        b = decoder.get(ch)
        if b is None:
            buf.extend(ch.encode("utf-8"))
        else:
            buf.append(b)
    return bytes(buf)


def _write_vocab_bin(
    vocab: list[str], out_path: Path, decoder: dict[str, int] | None
) -> None:
    with out_path.open("wb") as f:
        for token in vocab:
            # print(f"{token=}")
            token_bytes = _encode_token_to_bytes(token, decoder)
            if len(token_bytes) > 255:
                raise ValueError(
                    f"token byte length {len(token_bytes)} exceeds 255 limit"
                )
            f.write(bytes([len(token_bytes)]))
            f.write(token_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download tokenizer.json and generate vocab.bin."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model repo (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_TOKENIZER_FILENAME,
        help="Tokenizer filename in the repo (default: tokenizer.json)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"Output vocab file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--local-tokenizer-json",
        default=None,
        help="Use a local tokenizer.json instead of downloading",
    )
    byte_level_group = parser.add_mutually_exclusive_group()
    byte_level_group.add_argument(
        "--byte-level",
        action="store_true",
        help="Force byte-level token decoding (GPT-2/ByteLevel BPE style)",
    )
    byte_level_group.add_argument(
        "--no-byte-level",
        action="store_true",
        help="Disable byte-level token decoding",
    )
    args = parser.parse_args()

    if args.local_tokenizer_json:
        tokenizer_path = Path(args.local_tokenizer_json)
    else:
        tokenizer_path = _download_tokenizer_json(args.model, args.filename)

    data = _load_tokenizer_json(tokenizer_path)
    vocab = _build_vocab_list(data)

    if args.byte_level:
        decoder = _byte_decoder()
    elif args.no_byte_level:
        decoder = None
    else:
        decoder = _byte_decoder() if _has_byte_level_component(data) else None
        if decoder is None and _vocab_looks_byte_level(vocab):
            decoder = _byte_decoder()

    _write_vocab_bin(vocab, Path(args.out), decoder)
    print(f"wrote {len(vocab)} tokens to {args.out}")


if __name__ == "__main__":
    main()
