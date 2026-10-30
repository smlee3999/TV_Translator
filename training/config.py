"""
한↔영 양방향 번역기 하이퍼파라미터와 공통 상수.

단일 Transformer 하나로 양방향을 처리한다. 소스 문장 앞에 목표 언어 태그
(<2en> 또는 <2ko>)를 붙여 번역 방향을 지정한다.
    영→한: [<2ko>] + spm(en) + [</s>]  →  [<s>] + spm(ko) + [</s>]
    한→영: [<2en>] + spm(ko) + [</s>]  →  [<s>] + spm(en) + [</s>]
"""

# ---- SentencePiece (공유 vocab) ----
VOCAB_SIZE = 16000
PAD_ID = 0          # notebook 의 create_padding_mask 가 0 을 padding 으로 가정한다
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
TAG_TO_EN = "<2en>"
TAG_TO_KO = "<2ko>"
TAG_TO_EN_ID = 4    # user_defined_symbols 순서대로 4, 5 가 된다
TAG_TO_KO_ID = 5

# ---- 데이터 ----
MAX_TOKENS = 64          # 태그/BOS/EOS 포함 시퀀스 최대 길이 (positional encoding 길이와 같다)
MAX_CHARS = 200          # 정제 단계에서 이보다 긴 문장은 버린다
VALID_SIZE = 2000

# ---- 모델 (notebook 의 Transformer 생성자 인자) ----
NUM_LAYERS = 4
EMBEDDING_DIM = 256
NUM_HEADS = 4
FULLY_CONNECTED_DIM = 1024
DROPOUT_RATE = 0.1

# ---- 학습 ----
TOKENS_PER_BATCH = 8192  # 길이 버킷별로 batch_size = TOKENS_PER_BATCH // bucket_len
TRAIN_STEPS = 80000
WARMUP_STEPS = 4000
EVAL_EVERY = 2000
SAVE_EVERY = 1000


def model_kwargs(vocab_size=VOCAB_SIZE):
    return dict(num_layers=NUM_LAYERS,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                fully_connected_dim=FULLY_CONNECTED_DIM,
                input_vocab_size=vocab_size,
                target_vocab_size=vocab_size,
                max_positional_encoding_input=MAX_TOKENS,
                max_positional_encoding_target=MAX_TOKENS,
                dropout_rate=DROPOUT_RATE)
