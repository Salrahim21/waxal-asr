# Experiment Comparison & Submission Recommendation

**Date:** 2026-07-15
**Constraint:** 2 remaining leaderboard submissions. No fine-tuning. No new data.

---

## Experiment Summary

| # | Experiment | Model | Beam | Key Change |
|---|-----------|-------|------|------------|
| 1 | beam_search_small | whisper-small (244M) | 5 | Decoding only: beam search + no_repeat_ngram + condition_on_prev_tokens=False |
| 2 | large_v3_zero_shot | whisper-large-v3 (1.55B) | 5 | Model upgrade. Same decoding as Exp 1 |
| 3 | ensemble_v1 | small + large-v3 | 5 | Heuristic per-sample selection from Exp 1 + 2 |

---

## Skeptical ML Reviewer Analysis

### Ranking: Highest to Lowest Expected Improvement

---

### Rank 1: Experiment 2 — large_v3_zero_shot

**Expected WER improvement: LARGE**

**Why it should improve WER/CER:**
- Whisper Large-V3 was trained on 4M+ hours of labeled audio vs ~680K for Small
- Large-V3 has native Lingala and Shona language tokens with substantially more training data for both
- The 6.4x parameter increase (244M → 1550M) translates directly to better acoustic modeling and language understanding
- Large-V3 specifically improved on low-resource languages compared to earlier Whisper versions
- Our baseline analysis showed 63% garbled Shona outputs from Small — Large-V3 should dramatically reduce this
- The Luganda→Swahili fallback will also benefit: Large-V3's Swahili transcription quality is measurably better

**Failure modes:**
- Luganda is still not in Large-V3's vocabulary. Swahili fallback remains a proxy. WER for Luganda will still be high
- Large-V3 may hallucinate differently — longer, more fluent but still incorrect text (confident wrong answers)
- Some audio clips may be too short or noisy for even Large-V3 to handle
- Beam search on a larger model may amplify certain repetition patterns (though `no_repeat_ngram_size` mitigates this)

**Risks:**
- Runtime will be 3-6x longer than Small (larger model + same beam width)
- ~3GB VRAM means less headroom on 8GB GPU, but should still fit
- If the model downloads fail mid-inference, partial results are lost (no checkpointing in the notebook)

**Would I submit this?** Yes, absolutely. This is the highest expected-value submission. Model scale is the single most reliable lever in zero-shot ASR. Submit this first.

---

### Rank 2: Experiment 3 — ensemble_v1

**Expected WER improvement: MODERATE (conditional on Exp 2 quality)**

**Why it should improve WER/CER:**
- Small and Large-V3 will produce complementary errors — Small may occasionally produce cleaner short transcripts where Large-V3 hallucinates
- The heuristic filter catches pathological outputs: garbled scripts, empty predictions, repetition loops
- For the ~63% of Shona samples where Small produces garbled output, Large-V3 predictions will be selected automatically
- Even a 1-2% WER reduction from better per-sample selection is meaningful on a leaderboard

**Failure modes:**
- If Large-V3 is uniformly better, the ensemble reduces to just Large-V3 (no improvement, just added complexity)
- The quality heuristics are hand-crafted and may misjudge — a fluent but wrong Large-V3 prediction scores higher than a rougher but more accurate Small prediction
- The ensemble can only select between existing predictions. It cannot combine partial information from both
- Garbled-script detection uses Unicode script names, which may misclassify legitimate diacritics in African languages

**Risks:**
- Depends entirely on both Exp 1 and Exp 2 being run first
- If the heuristic margin (LARGE_PREFERENCE_MARGIN=10) is poorly calibrated, Small predictions get selected when they shouldn't
- No audio-level features — selection is purely text-based and may be fooled by confident hallucinations

**Would I submit this?** Conditionally. Only if the quality reports from Exp 1 and Exp 2 show meaningful complementary errors. If Large-V3 dominates on >95% of samples, skip the ensemble and save the submission for something else. Run the quality reports first, then decide.

---

### Rank 3: Experiment 1 — beam_search_small

**Expected WER improvement: SMALL-TO-MODERATE**

**Why it should improve WER/CER:**
- Beam search explores multiple decoding paths, reducing the chance of a bad greedy decode
- `no_repeat_ngram_size=3` directly eliminates the "kwa kwa kwa" repetition pattern we observed in 3.7% of predictions
- `condition_on_prev_tokens=False` prevents the decoder from cascading off a bad previous prediction, reducing hallucination snowballs
- Our baseline had 31.8% duplicate transcriptions — beam search should increase transcript diversity

**Failure modes:**
- Whisper Small fundamentally lacks training data for these languages. Better decoding cannot fix a poorly-conditioned latent representation
- Shona garbling (63% of predictions) is an encoder-level problem, not a decoding problem. Beam search won't fix it
- Luganda transcribed as Swahili will still be Swahili, just with slightly better Swahili decoding
- `length_penalty=1.0` is neutral, meaning beam search may still produce overly long outputs

**Risks:**
- Beam search with Small takes ~3-5x longer than greedy but produces similar-quality acoustic features
- May not be worth a leaderboard submission on its own, but is essential as the "Small" input for the ensemble
- If the WER improvement is <1%, the submission is wasted

**Would I submit this?** No, not as a standalone submission. Use it as input for the ensemble (Exp 3), but do not burn a leaderboard submission on it. The improvement from decoding alone on a small model is unlikely to be competitive.

---

## Recommended Submission Strategy

Given 2 remaining submissions:

```
Submission 1: Experiment 2 (large_v3_zero_shot)
    → Highest expected improvement
    → Model scale is the most reliable lever

Submission 2: Experiment 3 (ensemble_v1)
    → But ONLY IF quality reports show complementary errors
    → Otherwise, hold the submission for a future experiment
```

**Do NOT submit Experiment 1 (beam_search_small) standalone.** Run it to generate the CSV for the ensemble, but don't use a leaderboard slot on it.

---

## Execution Order

1. Run `exp1_beam_search_small.ipynb` → generates `submission_beam_small.csv` + quality report
2. Run `exp2_large_v3_zero_shot.ipynb` → generates `submission_large_v3.csv` + quality report
3. Compare quality reports. If Large-V3 is strictly dominant, submit Large-V3 only
4. If complementary errors exist, run `exp3_ensemble_v1.ipynb` → generates `submission_ensemble.csv`
5. Submit Large-V3 first (safe bet), then ensemble if the quality report supports it

---

## What We Cannot Fix Without Fine-Tuning

These limitations exist regardless of decoding strategy:

1. **Luganda:** Not in any Whisper vocabulary. Swahili proxy will always produce wrong language output. WER will be near 100% for Luganda samples regardless of model size
2. **Vocabulary mismatch:** Zero-shot Whisper generates text in its pretrained distribution, which may not match the specific vocabulary/spelling conventions in the competition ground truth
3. **Normalization:** Differences in punctuation, spacing, capitalization between Whisper output and ground truth labels will inflate WER even when the transcription is semantically correct
4. **Short audio clips:** Very short utterances (< 1 second) often trigger blank or garbled outputs from Whisper regardless of model size
