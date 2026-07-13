import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable
from functools import lru_cache

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab  # ID -> 字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()} # 字节块 -> ID
        
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)
        tokens = []
        last_pos = 0
        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            special_tok = match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            last_pos = match.end()
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))
        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)
        
        for p_tok in pre_tokens:
            ids.extend(self._encode_pre_token(p_tok))

        return ids

    @lru_cache(maxsize=100_000)
    def _encode_pre_token(self, p_tok: str) -> tuple[int, ...]:
        byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
        while len(byte_parts) >= 2:
            best_pair = None
            min_rank = float('inf')

            for i in range(len(byte_parts) - 1):
                pair = (byte_parts[i], byte_parts[i+1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            new_byte_parts = []
            i = 0
            while i < len(byte_parts):
                if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                    new_byte_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_byte_parts.append(byte_parts[i])
                    i += 1
            byte_parts = new_byte_parts

        return tuple(self.byte_to_id[part] for part in byte_parts)

    def decode(self, ids: list[int]) -> str:
        byte_segments = [self.id_to_byte[i] for i in ids]
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
