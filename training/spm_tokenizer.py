"""
텍스트 정규화와 SentencePiece unigram 토크나이저.

Android 앱(SpmTokenizer.kt)은 sentencepiece 네이티브 라이브러리 없이 vocab 파일만으로
토큰화를 한다. 여기의 UnigramTokenizer 는 그 Kotlin 구현과 같은 알고리즘의 Python
레퍼런스이며, sentencepiece 결과와 일치하는지 검증하는 데 사용한다.

학습 시 SentencePiece 는 normalization_rule_name='identity' 로 학습하고, 정규화는
normalize_text() 에서 직접 한다. 그래야 Kotlin 쪽에서 똑같이 재현할 수 있다.
"""
import re
import unicodedata

import numpy as np

import config

WS = "▁"  # '▁'
_WS_RE = re.compile(r"\s+")


def normalize_text(text):
    """NFKC + 공백 정리. Kotlin TextNormalizer.normalize 와 동일해야 한다."""
    text = unicodedata.normalize("NFKC", text)
    return _WS_RE.sub(" ", text).strip()


def export_vocab(sp, path):
    """id 순서대로 'piece<TAB>score<TAB>type' 을 기록한다. (score 는 float32 를 손실 없이)"""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            if sp.is_control(i):
                t = "c"
            elif sp.is_unknown(i):
                t = "u"
            elif sp.is_unused(i):
                t = "x"
            elif sp.is_byte(i):
                t = "b"
            elif sp.id_to_piece(i) in (config.TAG_TO_EN, config.TAG_TO_KO):
                t = "d"
            else:
                t = "n"
            f.write(f"{sp.id_to_piece(i)}\t{float(np.float32(sp.get_score(i))):.9g}\t{t}\n")


class UnigramTokenizer:
    """sentencepiece unigram 모델의 Viterbi 인코딩 (Kotlin SpmTokenizer 의 레퍼런스)."""

    UNK_PENALTY = np.float32(10.0)

    def __init__(self, vocab_path):
        self.pieces, self.scores, self.types = [], [], []
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                piece, score, t = line.rstrip("\n").split("\t")
                self.pieces.append(piece)
                self.scores.append(np.float32(score))
                self.types.append(t)
        self.piece_to_id = {p: i for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == "n"}
        normal = [s for s, t in zip(self.scores, self.types) if t == "n"]
        self.unk_score = np.float32(min(normal) - self.UNK_PENALTY)
        self.max_piece_len = max(len(p) for p in self.piece_to_id)
        # byte_fallback: '<0xEC>' 형태의 byte piece id
        self.byte_to_id = {int(p[3:5], 16): i for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == "b"}

    def encode(self, text):
        text = normalize_text(text)
        if not text:
            return []
        s = WS + text.replace(" ", WS)
        n = len(s)
        best = [np.float32(0)] + [None] * n
        back = [None] * (n + 1)  # (start, id)
        for start in range(n):
            if best[start] is None:
                continue
            has_single = False
            for length in range(1, min(self.max_piece_len, n - start) + 1):
                pid = self.piece_to_id.get(s[start:start + length])
                if pid is None:
                    continue
                if length == 1:
                    has_single = True
                cand = np.float32(best[start] + self.scores[pid])
                end = start + length
                if best[end] is None or cand > best[end]:
                    best[end], back[end] = cand, (start, pid)
            if not has_single:
                end = start + 1
                cand = np.float32(best[start] + self.unk_score)
                if best[end] is None or cand > best[end]:
                    best[end], back[end] = cand, (start, config.UNK_ID)
        segs, pos = [], n
        while pos > 0:
            start, pid = back[pos]
            segs.append((start, pos, pid))
            pos = start
        ids = []
        for start, end, pid in reversed(segs):
            if pid == config.UNK_ID and self.byte_to_id:
                ids.extend(self.byte_to_id[b] for b in s[start:end].encode("utf-8"))
            else:
                ids.append(pid)
        return ids

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            t = self.types[i]
            if t == "b":
                buf.append(int(self.pieces[i][3:5], 16))
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            if t == "n":
                out.append(self.pieces[i])
            elif t == "u":
                out.append(" \u2047 ")
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out).replace(WS, " ").strip()
