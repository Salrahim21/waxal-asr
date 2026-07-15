# Zero-Shot Whisper Small — Performance Report

**Date:** 2026-07-14
**Model:** `openai/whisper-small` (241.7M params, float16)
**Approach:** Zero-shot inference (no fine-tuning)
**Decoding:** Greedy (num_beams=1)

---

## Coverage

| Language | Test IDs | Predictions | Coverage | Missing |
|----------|----------|-------------|----------|---------|
| Luganda (lug) | 638 | 638 | 100.0% | 0 |
| Lingala (lin) | 1,866 | 1,832 | 98.2% | 34 |
| Shona (sna) | 1,749 | 1,596 | 91.3% | 153 |
| **Total** | **4,253** | **4,066** | **95.6%** | **187** |

> 187 IDs are missing because the `*-test-00000.parquet` files don't contain all test samples.
> Second shards (`lin-test-00001.parquet` ~8.5 MB, `sna-test-00001.parquet` ~47 MB) need to be downloaded to reach 100%.

---

## Quality Issues

| Issue | Count | % of predictions |
|-------|-------|-----------------|
| Likely English output | 2,651 | 65.2% |
| Repetitive hallucinations | 668 | 16.4% |
| `[BLANK_AUDIO]` tags | 96 | 2.4% |
| Duplicate transcriptions | 1,100 | 27.1% |

### The core problem: Whisper is transcribing in English

The ground truth is in **native African languages** (Luganda, Lingala, Shona), but Whisper zero-shot is outputting **English translations** instead of transcriptions. Examples:

| | Ground Truth (Train.csv) | Whisper Zero-Shot Output |
|---|---|---|
| **Luganda** | *Ekyuma ekyakolebwa Bamagulumeeru nga kiri mu makkati g'ennyanja...* | *I am not sure if I will be able to do this in September...* |
| **Lingala** | *Ndaku oyo ezali na bosoto mpo bazali kosala ngo likolo...* | *I will go to the new floor. I will go to the front...* |
| **Shona** | *Murume ari kufamba akapfeka hembe yekumusoro netirauzi...* | *I'm not a good person. I'm not a good person...* |

### Why this happens

Whisper has a `task` parameter: `"transcribe"` (output in source language) vs `"translate"` (output in English). Without explicit language/task settings, Whisper defaults to English translation for languages it's less confident about.

### Additional issues

- **Hallucinated repetitions:** 16.4% of outputs repeat the same phrase 3+ times (e.g., *"I am a dreamer. I am a dreamer. I am a dreamer..."*)
- **Length mismatch:** Whisper outputs average 350-450 chars while ground truth averages 150-210 chars
- **`[BLANK_AUDIO]`:** 96 samples produced no meaningful transcription

---

## Prediction vs Ground Truth Length

| Language | Avg Ground Truth Length | Avg Prediction Length | Ratio |
|----------|----------------------|----------------------|-------|
| Luganda | 208 chars | 404 chars | 1.9x too long |
| Lingala | 150 chars | 313 chars | 2.1x too long |
| Shona | 192 chars | 448 chars | 2.3x too long |

---

## Expected WER Score

With 65% of predictions in the wrong language entirely, the WER will be **very high (likely 90-100%)**. This submission serves as a baseline but will score near the bottom of the leaderboard.

---

## Recommended Fixes (Priority Order)

### 1. Force transcription mode with language hints (Quick fix, no training)

```python
model.generate(
    input_features,
    max_new_tokens=225,
    language=lang,        # e.g., "luganda", "lingala", "shona"
    task="transcribe",    # transcribe, NOT translate
)
```

This tells Whisper to output in the source language instead of translating to English. **This alone could dramatically improve WER.**

### 2. Download missing test shards (Quick fix, +187 predictions)

Download `lin-test-00001.parquet` (~8.5 MB) and `sna-test-00001.parquet` (~47 MB) to get 100% coverage.

### 3. Fine-tune on training data (Best results, requires ~5 GB download)

Fine-tuning Whisper Small on the 38K+ training examples would teach it the specific vocabulary and patterns of these languages. This is the approach most likely to produce a competitive WER score.

---

## Verdict

**This zero-shot submission is not competitive** due to Whisper outputting English instead of native language transcriptions. The immediate next step is to add `language` and `task="transcribe"` parameters to `model.generate()` — this requires zero additional data and should fix the core issue.
