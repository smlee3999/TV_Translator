"""
2단계: notebook 의 Transformer 로 한↔영 양방향 번역 모델을 학습한다.

- 각 문장쌍을 영→한, 한→영 두 개의 학습 예제로 만든다 (소스 앞에 목표 언어 태그).
- 길이 버킷으로 batch 를 만들어 padding 낭비를 줄인다.
- tf.train.CheckpointManager 로 주기적으로 저장하고, 다시 실행하면 이어서 학습한다.
  (Colab 세션이 끊겨도 Drive 의 checkpoint 에서 재개)

사용 예)
    python train.py --work_dir ./work
    python train.py --work_dir ./work --train_steps 2000 --eval_every 500   # 빠른 테스트
"""
import argparse
import os
import time
import warnings

import keras
import numpy as np
import sentencepiece as spm
import tensorflow as tf

import config
import matmul_grad
from spm_tokenizer import normalize_text
from transformer import Transformer, create_padding_mask, create_look_ahead_mask

# notebook 과 같이 Keras 3 의 mask 관련 경고는 무시한다
warnings.filterwarnings("ignore", message=".*does not support masking.*")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")

BUCKET_BOUNDARIES = [12, 16, 20, 24, 32, 40, 48]
# 각 버킷의 최대 길이. XLA 사용 시 batch 를 이 길이로 맞춰 shape 종류를 버킷 수(8개)로 고정한다.
BUCKET_LENGTHS = [b - 1 for b in BUCKET_BOUNDARIES] + [config.MAX_TOKENS]


def load_tsv(path, limit=None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            en, ko = line.rstrip("\n").split("\t")
            pairs.append((en, ko))
            if limit and len(pairs) >= limit:
                break
    return pairs


def encode_source(sp, text, to_lang):
    tag = config.TAG_TO_KO_ID if to_lang == "ko" else config.TAG_TO_EN_ID
    return [tag] + sp.encode(normalize_text(text)) + [config.EOS_ID]


def encode_pairs(sp, pairs):
    """(en, ko) 목록 → 양방향 예제의 (src, tgt) 패딩 배열. 너무 긴 예제는 버린다."""
    en_ids = sp.encode([p[0] for p in pairs])
    ko_ids = sp.encode([p[1] for p in pairs])
    srcs, tgts = [], []
    for e, k in zip(en_ids, ko_ids):
        for src_tag, src, tgt in ((config.TAG_TO_KO_ID, e, k), (config.TAG_TO_EN_ID, k, e)):
            s = [src_tag] + src + [config.EOS_ID]
            t = [config.BOS_ID] + tgt + [config.EOS_ID]
            if len(s) <= config.MAX_TOKENS and len(t) <= config.MAX_TOKENS:
                srcs.append(s)
                tgts.append(t)
    n = len(srcs)
    src_arr = np.zeros((n, config.MAX_TOKENS), np.int16)
    tgt_arr = np.zeros((n, config.MAX_TOKENS), np.int16)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_arr[i, :len(s)] = s
        tgt_arr[i, :len(t)] = t
    return src_arr, tgt_arr


def _nonpad_len(x, axis=None):
    return tf.reduce_sum(tf.cast(tf.not_equal(x, config.PAD_ID), tf.int32), axis=axis)


def make_dataset(src, tgt, tokens_per_batch, shuffle, fixed_shapes=False):
    """fixed_shapes=True 면 batch 크기와 길이를 버킷별로 고정한다 (XLA 재컴파일 방지)."""
    ds = tf.data.Dataset.from_tensor_slices((src, tgt))
    if shuffle:
        ds = ds.shuffle(200_000, reshuffle_each_iteration=True).repeat()

    def length(s, t):
        return tf.maximum(_nonpad_len(s), _nonpad_len(t))

    batch_sizes = [max(1, tokens_per_batch // b) for b in BUCKET_BOUNDARIES + [config.MAX_TOKENS]]
    ds = ds.bucket_by_sequence_length(length, BUCKET_BOUNDARIES, batch_sizes, drop_remainder=fixed_shapes)
    bucket_lengths = tf.constant(BUCKET_LENGTHS)

    def trim(s, t):
        n = tf.reduce_max(_nonpad_len(s, axis=1))
        m = tf.reduce_max(_nonpad_len(t, axis=1))
        if fixed_shapes:
            n = m = bucket_lengths[tf.reduce_sum(tf.cast(bucket_lengths < tf.maximum(n, m), tf.int32))]
        return tf.cast(s[:, :n], tf.int32), tf.cast(t[:, :m], tf.int32)

    return ds.map(trim, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Attention Is All You Need 의 warmup 학습률."""

    def __init__(self, d_model, warmup_steps):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32) + 1.0
        return tf.math.rsqrt(self.d_model) * tf.minimum(tf.math.rsqrt(step), step * self.warmup_steps ** -1.5)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


def build_model(vocab_size, mixed_precision=False):
    """mixed_precision=True 면 fp16 으로 계산한다 (가중치는 float32 그대로라 checkpoint 호환).
    notebook 코드는 임베딩 출력에 float32 positional encoding 을 더하므로 임베딩과
    softmax 출력층은 float32 로 둔다."""
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = Transformer(**config.model_kwargs(vocab_size))
    finally:
        keras.mixed_precision.set_global_policy("float32")
    if mixed_precision:
        for layer in (model.encoder.embedding, model.decoder.embedding, model.final_layer):
            layer.dtype_policy = "float32"
    dummy = tf.constant([[config.TAG_TO_KO_ID, 7, config.EOS_ID]], tf.int32)
    model(dummy, dummy, training=False)
    return model


def make_checkpoint(model, **objects):
    """Keras 3 는 enc_layers/dec_layers 처럼 list 에 담긴 레이어를 tf.train.Checkpoint 에 노출하지 않아
    model 만 넘기면 인코더/디코더 레이어가 복원되지 않는다. 그래서 가중치를 직접 등록한다."""
    return tf.train.Checkpoint(model=model, weights=list(model.trainable_variables), **objects)


def restore_checkpoint(ckpt, model, path):
    """make_checkpoint 로 만든 ckpt 를 복원한다. weights 가 없는 이전 형식 checkpoint 는 가중치가
    optimizer/_trainable_variables/N 에 저장되어 있으므로 거기서 직접 읽는다 (N 은 trainable_variables 순서)."""
    status = ckpt.restore(path)
    reader = tf.train.load_checkpoint(path)
    keys = reader.get_variable_to_shape_map()
    variables = model.trainable_variables
    attr = "/.ATTRIBUTES/VARIABLE_VALUE"
    if all(f"weights/{i}{attr}" in keys for i in range(len(variables))):
        return status
    legacy = [f"optimizer/_trainable_variables/{i}{attr}" for i in range(len(variables))]
    # 이전 형식에서 optimizer 쪽에 없는 가중치는 model/ 경로로 저장되어 위 restore 에서 복원된다
    in_model = sum(k.startswith("model/") and k.endswith(attr) for k in keys)
    if sum(k in keys for k in legacy) + in_model != len(variables):
        raise ValueError(f"{path} 에서 모델 가중치 {len(variables)}개를 모두 찾을 수 없습니다.")
    for key, v in zip(legacy, variables):
        if key in keys:
            v.assign(reader.get_tensor(key))
    print(f"이전 형식 checkpoint 에서 가중치 복구: {path}")
    return status


def masked_loss_and_acc(real, pred):
    mask = tf.cast(tf.not_equal(real, config.PAD_ID), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    hit = tf.cast(tf.equal(real, tf.argmax(pred, axis=-1, output_type=tf.int32)), tf.float32)
    acc = tf.reduce_sum(hit * mask) / tf.reduce_sum(mask)
    return loss, acc


def forward(model, src, tgt, training):
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    enc_mask = create_padding_mask(src)
    look_ahead = create_look_ahead_mask(tf.shape(tgt_in)[1])
    pred, _ = model(src, tgt_in, training=training, enc_padding_mask=enc_mask,
                    look_ahead_mask=look_ahead, dec_padding_mask=enc_mask)
    return masked_loss_and_acc(tgt_out, pred)


def max_output_len(src_len):
    """반복 생성이 길게 이어지지 않도록 출력 길이를 소스 길이에 비례해 제한한다 (앱과 동일)."""
    return min(config.MAX_TOKENS, 2 * src_len + 10)


def greedy_ids(model, src_ids):
    """Keras 모델로 greedy decoding. src_ids 는 [태그] + 토큰 + [EOS]."""
    src = tf.constant([src_ids], tf.int32)
    enc_mask = create_padding_mask(src)
    enc_out = model.encoder(src, training=False, mask=enc_mask)
    out = [config.BOS_ID]
    while len(out) < max_output_len(len(src_ids)):
        dec, _ = model.decoder(tf.constant([out], tf.int32), enc_out, training=False,
                               look_ahead_mask=create_look_ahead_mask(len(out)), padding_mask=enc_mask)
        next_id = int(tf.argmax(model.final_layer(dec[:, -1, :]), axis=-1)[0])
        if next_id == config.EOS_ID:
            break
        out.append(next_id)
    return out[1:]


def greedy_translate(model, sp, text, to_lang):
    src = encode_source(sp, text, to_lang)
    src = src[:config.MAX_TOKENS - 1] + [config.EOS_ID] if len(src) > config.MAX_TOKENS else src
    return sp.decode(greedy_ids(model, src))


def main(work_dir, train_steps=config.TRAIN_STEPS, eval_every=config.EVAL_EVERY,
         save_every=config.SAVE_EVERY, tokens_per_batch=config.TOKENS_PER_BATCH,
         max_pairs=None, mixed_precision=None, xla=None):
    """mixed_precision / xla 는 None 이면 GPU 가 있을 때만 켠다."""
    sp = spm.SentencePieceProcessor(model_file=os.path.join(work_dir, "spm.model"))
    vocab_size = sp.get_piece_size()
    matmul_grad.install()  # 배치 크기만큼 가중치 gradient 를 복제하는 OOM 원인 제거
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    mixed_precision = has_gpu if mixed_precision is None else mixed_precision
    xla = has_gpu if xla is None else xla
    print(f"GPU {has_gpu}, mixed_precision {mixed_precision}, XLA {xla}, tokens_per_batch {tokens_per_batch}")

    t0 = time.time()
    train_pairs = load_tsv(os.path.join(work_dir, "train.tsv"), max_pairs)
    valid_pairs = load_tsv(os.path.join(work_dir, "valid.tsv"))
    tr_src, tr_tgt = encode_pairs(sp, train_pairs)
    va_src, va_tgt = encode_pairs(sp, valid_pairs)
    print(f"train examples {len(tr_src):,} (양방향), valid {len(va_src):,}  [{time.time() - t0:.0f}s]")
    train_ds = make_dataset(tr_src, tr_tgt, tokens_per_batch, shuffle=True, fixed_shapes=xla)
    valid_ds = make_dataset(va_src, va_tgt, tokens_per_batch, shuffle=False)

    model = build_model(vocab_size, mixed_precision)
    adam = tf.keras.optimizers.Adam(NoamSchedule(config.EMBEDDING_DIM, config.WARMUP_STEPS),
                                    beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # fp16 은 gradient underflow 를 막기 위해 loss scaling 이 필요하다
    optimizer = keras.optimizers.LossScaleOptimizer(adam) if mixed_precision else adam
    step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    # Adam 자체를 저장해야 mixed_precision 설정과 관계없이 checkpoint 를 이어 쓸 수 있다
    ckpt = make_checkpoint(model, optimizer=adam, step=step_var)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(work_dir, "ckpt"), max_to_keep=3)
    if manager.latest_checkpoint:
        restore_checkpoint(ckpt, model, manager.latest_checkpoint)
        print(f"checkpoint 복원: {manager.latest_checkpoint} (step {int(step_var.numpy())})")
    print(f"parameters: {model.count_params():,}")

    spec = tf.TensorSpec([None, None], tf.int32)

    @tf.function(input_signature=[spec, spec], jit_compile=xla)
    def train_step(src, tgt):
        with tf.GradientTape() as tape:
            loss, acc = forward(model, src, tgt, training=True)
            scaled_loss = optimizer.scale_loss(loss) if mixed_precision else loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        step_var.assign_add(1)
        return loss, acc

    @tf.function(input_signature=[spec, spec])
    def eval_step(src, tgt):
        return forward(model, src, tgt, training=False)

    samples = [("en", "ko", valid_pairs[0][0]), ("ko", "en", valid_pairs[0][1]),
               ("en", "ko", "How are you today?"), ("ko", "en", "오늘 날씨가 정말 좋네요.")]

    loss_m, acc_m = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    t0 = time.time()
    step = int(step_var.numpy())  # 매 step 마다 GPU 값을 읽으면 동기화로 느려지므로 Python 에서 센다
    for src, tgt in train_ds:
        if step >= train_steps:
            break
        loss, acc = train_step(src, tgt)
        loss_m.update_state(loss)
        acc_m.update_state(acc)
        step += 1
        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss_m.result():.4f}  acc {acc_m.result():.4f}  "
                  f"lr {float(optimizer.learning_rate):.2e}  {(time.time() - t0) / 100:.3f}s/step", flush=True)
            loss_m.reset_state()
            acc_m.reset_state()
            t0 = time.time()
        if step % save_every == 0 or step == train_steps:
            manager.save(checkpoint_number=step)
        if step % eval_every == 0 or step == train_steps:
            vl, va = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
            for vs, vt in valid_ds:
                l, a = eval_step(vs, vt)
                vl.update_state(l)
                va.update_state(a)
            print(f"== step {step}  valid loss {vl.result():.4f}  valid acc {va.result():.4f}")
            for src_lang, tgt_lang, text in samples:
                print(f"   [{src_lang}→{tgt_lang}] {text}\n            → {greedy_translate(model, sp, text, tgt_lang)}")
            t0 = time.time()
    manager.save(checkpoint_number=int(step_var.numpy()))
    return model, sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--train_steps", type=int, default=config.TRAIN_STEPS)
    ap.add_argument("--eval_every", type=int, default=config.EVAL_EVERY)
    ap.add_argument("--save_every", type=int, default=config.SAVE_EVERY)
    ap.add_argument("--tokens_per_batch", type=int, default=config.TOKENS_PER_BATCH)
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()
    main(args.work_dir, args.train_steps, args.eval_every, args.save_every,
         args.tokens_per_batch, args.max_pairs)
