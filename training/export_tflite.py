"""
3단계: 학습된 checkpoint 를 Android 용 TFLite 모델로 변환하고 앱 assets 를 만든다.

TFLite 모델은 두 개의 signature 를 가진다 (batch=1, 시퀀스 길이는 가변).
    encode(src_ids[1, S] int32)                         -> enc_output[1, S, D] float32
    decode(enc_output[1, S, D] float32, tgt_ids[1, T])  -> next_id[1] int32
decode 는 tgt_ids 의 마지막 위치에 대해서만 final_layer 를 계산하고 argmax 를 돌려준다.
앱은 decode 를 반복 호출하는 greedy decoding 을 한다 (TFLiteTranslator 와 같은 로직).

생성되는 파일 (out_dir):
    translator.tflite         모델
    tokenizer.vocab           SentencePiece vocab (piece<TAB>score<TAB>type)
    model_config.json         특수 토큰 id, 최대 길이 등
    tokenizer_testcases.tsv   Kotlin 토크나이저 단위 테스트용 (text<TAB>ids)

사용 예)
    python export_tflite.py --work_dir ./work --android_dir ../android
"""
import argparse
import json
import os
import re
import shutil
import time
import warnings

import keras
import numpy as np
import tensorflow as tf

import config
from spm_tokenizer import UnigramTokenizer
from train import build_model, encode_source, greedy_ids, load_tsv, max_output_len
from transformer import create_look_ahead_mask

warnings.filterwarnings("ignore", message=".*does not support masking.*")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")

ASSET_FILES = ("translator.tflite", "tokenizer.vocab", "model_config.json")


def export_saved_model(model, path):
    """batch=1 추론 전용 encode/decode signature 로 SavedModel 을 만든다.
    패딩이 없으므로 padding mask 는 쓰지 않는다."""

    def encode(src_ids):
        return {"enc_output": model.encoder(src_ids, training=False, mask=None)}

    def decode(enc_output, tgt_ids):
        look_ahead = create_look_ahead_mask(tf.shape(tgt_ids)[1])
        dec, _ = model.decoder(tgt_ids, enc_output, training=False,
                               look_ahead_mask=look_ahead, padding_mask=None)
        probs = model.final_layer(dec[:, -1, :])
        return {"next_id": tf.argmax(probs, axis=-1, output_type=tf.int32)}

    # Keras 3 의 ExportArchive 가 모델의 모든 변수를 명시적으로 추적해 준다.
    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint("encode", encode, input_signature=[
        tf.TensorSpec([1, None], tf.int32, name="src_ids")])
    archive.add_endpoint("decode", decode, input_signature=[
        tf.TensorSpec([1, None, config.EMBEDDING_DIM], tf.float32, name="enc_output"),
        tf.TensorSpec([1, None], tf.int32, name="tgt_ids")])
    archive.write_out(path)


def convert(saved_model_dir, quantize):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=["encode", "decode"])
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range: 가중치 int8, 약 1/4 크기
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()


class TFLiteTranslator:
    """Android 앱의 Translator.kt 와 같은 로직 (문장 분리 → 토큰화 → greedy decoding)."""

    SENTENCE_SPLIT = re.compile(r"(?<=[.!?。？！])\s+|\n+")

    def __init__(self, model_content, vocab_path):
        self.interp = tf.lite.Interpreter(model_content=model_content, num_threads=4)
        self.encode_fn = self.interp.get_signature_runner("encode")
        self.decode_fn = self.interp.get_signature_runner("decode")
        self.tok = UnigramTokenizer(vocab_path)

    def translate_ids(self, src_ids):
        enc = self.encode_fn(src_ids=np.array([src_ids], np.int32))["enc_output"]
        out = [config.BOS_ID]
        while len(out) < max_output_len(len(src_ids)):
            nid = int(self.decode_fn(enc_output=enc, tgt_ids=np.array([out], np.int32))["next_id"][0])
            if nid == config.EOS_ID:
                break
            out.append(nid)
        return out[1:]

    def translate(self, text, to_lang):
        tag = config.TAG_TO_KO_ID if to_lang == "ko" else config.TAG_TO_EN_ID
        results = []
        for sentence in self.SENTENCE_SPLIT.split(text):
            ids = self.tok.encode(sentence)
            chunk = config.MAX_TOKENS - 2  # 태그와 EOS 자리
            for i in range(0, len(ids), chunk):
                results.append(self.tok.decode(self.translate_ids([tag] + ids[i:i + chunk] + [config.EOS_ID])))
        return " ".join(r for r in results if r)


def write_tokenizer_testcases(tok, texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\t", " ").replace("\n", " ") + "\t" + " ".join(map(str, tok.encode(t))) + "\n")


def main(work_dir, out_dir=None, android_dir=None, quantize=True, verify_n=50):
    out_dir = out_dir or os.path.join(work_dir, "android_assets")
    os.makedirs(out_dir, exist_ok=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(work_dir, "spm.model"))

    model = build_model(sp.get_piece_size())
    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(os.path.join(work_dir, "ckpt"))
    assert latest, "checkpoint 가 없습니다. train.py 를 먼저 실행하세요."
    ckpt.restore(latest).expect_partial()
    print(f"restored {latest}")

    sm_dir = os.path.join(work_dir, "saved_model")
    export_saved_model(model, sm_dir)

    tflite = convert(sm_dir, quantize)
    tflite_path = os.path.join(out_dir, "translator.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite)
    print(f"TFLite: {tflite_path} ({len(tflite) / 1e6:.1f} MB, quantize={quantize})")

    vocab_path = os.path.join(out_dir, "tokenizer.vocab")
    shutil.copy(os.path.join(work_dir, "tokenizer.vocab"), vocab_path)
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump({"pad_id": config.PAD_ID, "unk_id": config.UNK_ID, "bos_id": config.BOS_ID,
                   "eos_id": config.EOS_ID, "to_en_id": config.TAG_TO_EN_ID, "to_ko_id": config.TAG_TO_KO_ID,
                   "max_tokens": config.MAX_TOKENS, "embedding_dim": config.EMBEDDING_DIM,
                   "vocab_size": sp.get_piece_size()}, f, indent=2)

    # ---- 검증: 같은 입력에 대해 TFLite greedy 결과가 Keras greedy 결과와 같은지 ----
    valid = load_tsv(os.path.join(work_dir, "valid.tsv"), verify_n)
    tl = TFLiteTranslator(tflite, vocab_path)
    same, total, t_sum = 0, 0, 0.0
    for en, ko in valid:
        for text, to_lang in ((en, "ko"), (ko, "en")):
            src = encode_source(sp, text, to_lang)
            if len(src) > config.MAX_TOKENS:
                continue
            ref = greedy_ids(model, src)
            t0 = time.time()
            got = tl.translate_ids(src)
            t_sum += time.time() - t0
            same += int(ref == got)
            total += 1
            if total <= 6:
                print(f"  [{to_lang}] {text}\n      keras : {sp.decode(ref)}\n      tflite: {tl.tok.decode(got)}")
    print(f"TFLite vs Keras 토큰 일치 {same}/{total}  (TFLite 평균 {t_sum / max(total, 1) * 1000:.0f} ms/문장, 이 PC 기준)")

    tok = UnigramTokenizer(vocab_path)
    cases = [t for p in load_tsv(os.path.join(work_dir, "valid.tsv"), 300) for t in p]
    cases += ["  hello   world  ", "Ｈｅｌｌｏ　ｗｏｒｌｄ", "안녕하세요 😀 世界", "샴페인 솅겐 뎀프시", "ㄱㄴㄷ 쀍뷁", "a b"]
    write_tokenizer_testcases(tok, cases, os.path.join(out_dir, "tokenizer_testcases.tsv"))

    if android_dir:
        assets = os.path.join(android_dir, "app", "src", "main", "assets")
        test_res = os.path.join(android_dir, "app", "src", "test", "resources")
        os.makedirs(assets, exist_ok=True)
        os.makedirs(test_res, exist_ok=True)
        for name in ASSET_FILES:
            shutil.copy(os.path.join(out_dir, name), os.path.join(assets, name))
        for name in ("tokenizer.vocab", "tokenizer_testcases.tsv"):
            shutil.copy(os.path.join(out_dir, name), os.path.join(test_res, name))
        print(f"Android 프로젝트에 복사 완료: {assets}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--android_dir", default=None, help="Android 프로젝트 루트 (assets 자동 복사)")
    ap.add_argument("--no_quantize", action="store_true")
    ap.add_argument("--verify_n", type=int, default=50)
    args = ap.parse_args()
    main(args.work_dir, args.out_dir, args.android_dir, not args.no_quantize, args.verify_n)
