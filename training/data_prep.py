"""
1단계: 한·영 병렬 코퍼스 다운로드/정제 + 공유 SentencePiece vocab 학습.

기본 코퍼스는 OPUS-100 en-ko (Hugging Face: Helsinki-NLP/opus-100, 학습 100만 쌍).
AI Hub 등 다른 코퍼스가 있으면 'en<TAB>ko' 형식 TSV 로 만들어 extra_tsv 로 넘기면 함께 사용한다.

사용 예)
    python data_prep.py --work_dir ./work
    python data_prep.py --work_dir ./work --extra_tsv aihub_enko.tsv
"""
import argparse
import os
import random
import re
import urllib.request

import pandas as pd
import sentencepiece as spm

import config
from spm_tokenizer import normalize_text, export_vocab

OPUS100_URL = "https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-ko/{split}-00000-of-00001.parquet"

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def download_opus100(work_dir):
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(raw_dir, f"opus100_{split}.parquet")
        if not os.path.exists(path):
            print(f"downloading OPUS-100 en-ko {split} ...")
            urllib.request.urlretrieve(OPUS100_URL.format(split=split), path)
        paths[split] = path
    return paths


def read_opus100(path):
    df = pd.read_parquet(path)
    return [(t["en"], t["ko"]) for t in df["translation"]]


def read_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2:
                pairs.append((cols[0], cols[1]))
    return pairs


def clean_pair(en, ko):
    """정규화 후 품질이 낮은 쌍은 None 을 돌려준다."""
    en, ko = normalize_text(en), normalize_text(ko)
    if not en or not ko or en == ko:
        return None
    if len(en) > config.MAX_CHARS or len(ko) > config.MAX_CHARS:
        return None
    if not _HANGUL.search(ko) or _HANGUL.search(en) or not _LATIN.search(en):
        return None
    ratio = len(ko) / len(en)  # 한국어는 보통 영어 글자 수의 0.3~1.0 배
    if ratio < 0.15 or ratio > 2.5:
        return None
    return en, ko


def clean_pairs(pairs):
    seen, out = set(), []
    for en, ko in pairs:
        p = clean_pair(en, ko)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def write_tsv(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for en, ko in pairs:
            f.write(f"{en}\t{ko}\n")


def train_sentencepiece(train_pairs, work_dir, vocab_size=config.VOCAB_SIZE):
    text_path = os.path.join(work_dir, "spm_input.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for en, ko in train_pairs:
            f.write(en + "\n" + ko + "\n")
    prefix = os.path.join(work_dir, "spm")
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        byte_fallback=True,  # vocab 에 없는 드문 글자(인명 등)는 UTF-8 바이트 토큰으로 표현
        normalization_rule_name="identity",  # 정규화는 normalize_text 에서 직접 한다
        pad_id=config.PAD_ID, unk_id=config.UNK_ID, bos_id=config.BOS_ID, eos_id=config.EOS_ID,
        user_defined_symbols=[config.TAG_TO_EN, config.TAG_TO_KO],
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
        minloglevel=1,
    )
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    assert sp.piece_to_id(config.TAG_TO_EN) == config.TAG_TO_EN_ID
    assert sp.piece_to_id(config.TAG_TO_KO) == config.TAG_TO_KO_ID
    export_vocab(sp, os.path.join(work_dir, "tokenizer.vocab"))
    os.remove(text_path)
    return sp


def main(work_dir, extra_tsv=(), max_pairs=None, vocab_size=config.VOCAB_SIZE, seed=42):
    os.makedirs(work_dir, exist_ok=True)
    raw = download_opus100(work_dir)
    train = read_opus100(raw["train"])
    for path in extra_tsv:
        extra = read_tsv(path)
        print(f"extra corpus {path}: {len(extra):,} pairs")
        train += extra
    print(f"raw train pairs: {len(train):,}")
    train = clean_pairs(train)
    print(f"clean train pairs: {len(train):,}")

    held_out = clean_pairs(read_opus100(raw["validation"]) + read_opus100(raw["test"]))
    held_out_set = set(held_out)
    train = [p for p in train if p not in held_out_set]
    random.Random(seed).shuffle(train)
    if max_pairs:
        train = train[:max_pairs]
    valid, test = held_out[:config.VALID_SIZE], held_out[config.VALID_SIZE:]

    write_tsv(train, os.path.join(work_dir, "train.tsv"))
    write_tsv(valid, os.path.join(work_dir, "valid.tsv"))
    write_tsv(test, os.path.join(work_dir, "test.tsv"))
    print(f"train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

    sp = train_sentencepiece(train, work_dir, vocab_size)
    print(f"SentencePiece vocab: {sp.get_piece_size()}")
    for en, ko in valid[:3]:
        print(" ", sp.encode(en, out_type=str))
        print(" ", sp.encode(ko, out_type=str))
    return sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--extra_tsv", nargs="*", default=[])
    ap.add_argument("--max_pairs", type=int, default=None, help="빠른 테스트용으로 학습 쌍 수 제한")
    ap.add_argument("--vocab_size", type=int, default=config.VOCAB_SIZE)
    args = ap.parse_args()
    main(args.work_dir, args.extra_tsv, args.max_pairs, args.vocab_size)
