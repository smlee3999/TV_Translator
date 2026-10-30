# Android TV 한↔영 번역기

`coursera_transformer_jnb/C5_W4_A1_Transformer_Subclass_v1.ipynb` 에서 구현한 Transformer 로
한↔영 **양방향** 번역 모델을 학습하고, Android TV 앱에서 기기 내(오프라인) 추론으로 실행한다.

```
coursera_transformer_jnb/  원본 Coursera 노트북 (수정하지 않음)
training/                  학습 · 변환 파이프라인 (Python)
android/                   Android TV 앱 (Kotlin, LiteRT)
```

## 전체 흐름

1. `training/` 폴더를 Google Drive 에 올리고 `training/KoEn_Translator_Colab.ipynb` 를 Colab(GPU)에서 실행한다.
2. 마지막 셀에서 내려받은 `android_assets.zip` 의 파일을 복사한다.
   - `translator.tflite`, `tokenizer.vocab`, `model_config.json` → `android/app/src/main/assets/`
   - `tokenizer.vocab`, `tokenizer_testcases.tsv` → `android/app/src/test/resources/`
3. `android/` 를 Android Studio 로 열어 빌드한 뒤 TV 에 설치한다 (`./gradlew installDebug`).

## 모델 설계

- **단일 모델 + 방향 태그**: 소스 앞에 목표 언어 태그를 붙여 한 모델로 양방향 번역을 한다.
  - 영→한: `[<2ko>] + en + [</s>]` → `[<s>] + ko + [</s>]`
  - 한→영: `[<2en>] + ko + [</s>]` → `[<s>] + en + [</s>]`
- **Transformer**: 노트북의 `Transformer` 클래스를 그대로 쓴다(`training/transformer.py`, 단위 테스트 8개 통과).
  4 layers, d_model 256, 4 heads, FFN 1024, 파라미터 약 2,900만 개 (`training/config.py`).
- **토크나이저**: 한·영 공유 SentencePiece unigram 16k, `byte_fallback` 으로 드문 한글 음절(인명 등)도 표현한다.
- **데이터**: OPUS-100 en-ko (100만 쌍, 자막 위주) → 정제 후 약 82만 쌍 × 양방향.
  AI Hub 같은 코퍼스는 `en<TAB>ko` TSV 로 만들어 `data_prep.main(..., extra_tsv=[...])` 에 넘긴다.
- **TFLite**: `encode` / `decode` 두 signature, 가변 길이, int8 동적 양자화 (약 29MB).
  `decode` 는 마지막 위치의 다음 토큰 id 만 돌려주고, 앱이 greedy decoding 을 반복한다.

## Android 앱

- 리모컨: D-pad 로 이동하고, 입력창에서 가운데 버튼을 누르면 화상 키보드가 열린다. '완료'를 누르면 번역한다.
- 블루투스/USB 키보드: Enter 로 번역한다. **앱 내장 두벌식 한글 조합기**를 쓰며, 한/영 키 · 오른쪽 Alt · Shift+Space 또는 화면의 `키보드: 한/A` 버튼으로 전환한다.
  (한글 조합을 지원하지 않는 TV IME 가 많아서 앱이 직접 조합한다)
- 번역 방향: 자동 감지(한글이 있으면 한→영) / 한→영 / 영→한
- 긴 입력은 문장 단위(. ! ? 줄바꿈)로 나눠서 번역한다.

## 검증한 내용

| 항목 | 결과 |
|---|---|
| 노트북 단위 테스트 (`transformer.py`) | 8/8 통과 |
| Python 레퍼런스 토크나이저 vs sentencepiece | 47,549 문장 ID/디코딩 완전 일치 |
| Kotlin 토크나이저 vs sentencepiece (`SpmTokenizerTest`) | 606 케이스 일치 |
| 두벌식 조합기 (`HangulComposerTest`) | 6개 테스트 통과 |
| TFLite(float32) vs Keras greedy 결과 | 20/20 토큰 단위 일치 |
| 에뮬레이터 실행 | 모델 로딩, 번역, 화상 키보드 '완료', D-pad 포커스 동작 확인 |

로컬 검증은 CPU 에서 200 step 만 학습한 모델로 했기 때문에 번역 품질은 확인하지 않았다. 실제 품질은 Colab 학습 후 확인해야 한다.

## 참고 / 한계

- OPUS-100 은 영화 자막 중심이라 구어체에 강하고, 뉴스·기술 문서에는 약하다. 품질을 높이려면 AI Hub 한영 코퍼스를 추가하는 것이 가장 효과적이다.
- 노트북 구현은 `MultiHeadAttention(key_dim=embedding_dim)` 으로 헤드당 차원이 d_model 이라 일반적인 구현보다 attention 연산량이 4배 크다. 노트북 코드를 그대로 쓰기 위해 바꾸지 않았다.
- 학습 환경을 바꿀 때는 TensorFlow 와 Keras 버전을 짝에 맞게 써야 한다 (예: TF 2.16 ↔ Keras 3.3).
