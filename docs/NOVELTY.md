# AgriSmart: novel contributions (input for the novelty report)

All numbers come from `experiments/EXPERIMENTS_SUMMARY.md`, which `ml/train_models.py` and `ml/experiments.py` regenerate. Every model is trained on the 80% split and tested on the 20% split (seed 42, 440 test rows). Deployed models: **Soft Voting (RF+HGB+SVM)** with pH, **Random Forest** without pH, both isotonic-calibrated (see Exp 7).

## Problem with existing crop recommenders (from the Review-1 gaps)
1. They need an **N-P-K sensor** (₹2,500+, RS485, 12 V supply), or a lab test for every recommendation.
2. They classify **one snapshot** of soil values, but a cheap sensor's single reading is noisy, swings with the day/night cycle, and sometimes faults (e.g., a DS18B20 returning -127 °C).
3. They always output an answer, **even when unsure** or when the input is nothing like the training data.
4. They ignore the **plot's history**: the recommendation is the same whether the plot just grew that crop or not, and whether its nitrogen is depleted or not.

## N1: Quality-Weighted Temporal Consensus (QW-TC)
The database already stores a time series per plot. Instead of classifying the latest row, the engine builds a consensus feature vector over the last 7 days:

`x̄ = Σ qᵢ·wᵢ·xᵢ / Σ qᵢ·wᵢ`

- `qᵢ` is a data-quality score from four checks:
  - a physical-range check;
  - known sensor fault codes (-127 / 85 °C);
  - a robust spike test (MAD z-score against the window median);
  - soil-air temperature consistency.
- `wᵢ` is a **day-blocked** recency weight with τ = 7 days.

**Exp 3:** 42 simulated readings per test row with diurnal cycles, datasheet-level noise, and injected faults at rate f.

| fault rate | snapshot (conventional) | mean of per-reading probabilities | **QW-TC (proposed)** |
|---|---|---|---|
| 0% | 0.580 | 0.809 | **0.902** |
| 10% | 0.589 | 0.793 | **0.875** |
| 30% | 0.539 | 0.823 | **0.857** |

Two findings from the ablations:

1. **Fuse features, not probabilities.** Averaging the per-reading class probabilities (method D) is clearly worse than averaging the features first. Single readings carry the day/night swing, while the crop's requirements describe daily-mean conditions.
2. **Use day-blocked recency, not hourly decay.** An hourly recency decay (τ = 72 h, method G) lowered accuracy because it over-weights the most recent hours. Day-blocked weighting keeps the day/night balance.

Quality weighting matters most at high fault rates. At 30% faults, quality weighting without recency reaches 0.857, against 0.848 for plain range-filtering.

Differences under about 3 points are within sampling noise (SE ≈ 0.017).

## N2: Feature-availability adaptive model selection
pH is optional on the node. If a valid pH (sensor or manual, at most 180 days old) exists, the *with-pH* model is used; otherwise the *sensor-only* model is used.

**Exp 4** (no pH available):

| strategy | accuracy |
|---|---|
| with-pH model, pH imputed with the training median | 0.791 |
| with-pH model, pH assumed 7.0 | 0.677 |
| **switch to sensor-only model (proposed)** | **0.814** |
| reference: pH actually measured | 0.902 |

## N3: Calibrated, abstention-aware recommendations
- **Isotonic calibration** reduced expected calibration error from 0.086 to 0.022 (with pH, Soft Voting) and from 0.074 to 0.051 (without pH, Random Forest), so a "70%" really means about 70%.
- An **Isolation-Forest** detector marks readings unlike any training condition. It flagged 70.1% of random in-range readings but only 0.45% of real test readings.
- A **confidence threshold** of 0.5, chosen from the risk-coverage curve, keeps 97.0% of recommendations, which are 91.8% accurate. The flagged 3.0% are only about 38% accurate, so the UI tells the user to get a lab test instead.

## N4: Virtual Nitrogen Balance Index (VNBI) and context-aware re-ranking
With no NPK sensor, nitrogen status is **estimated from the database**:

`VNBI = Σ (k = last 3 cycles) 0.6^k · ( effect(crop_k) + min(1, N applied per acre / 40) )`

Here `effect` is +1 for a legume (fixer), 0 for low, −0.5 for medium and −1 for a high N-demand crop. The index is implemented identically in SQL (`fn_virtual_n_index`) and Python.

**Final score** = calibrated probability × season fit × rotation fit × nitrogen fit:

- **Season fit:** off-season ×0.6.
- **Rotation fit:**
  - same annual crop ×0 (also enforced by the trigger);
  - same family ×0.8;
  - legume after a cereal or fibre crop ×1.15.
- **Nitrogen fit:**
  - when VNBI < −0.75 (likely N-depleted): high-N crop ×0.85, legume ×1.15;
  - when VNBI > 0.75 (likely N-rich): high-N crop ×1.1.

**Exp 6** (one real reading, five histories): the model alone ranks maize 0.62, lentil 0.38. In July with no history, the off-season lentil drops to 0.23.

- When the last crop was maize, maize is blocked and lentil (a legume) moves to #1 (0.30).
- After three N-hungry crops with no fertiliser (VNBI −1.96), maize falls from 0.62 to 0.52.
- When the same history had 50 kg N/acre urea applied each cycle, the penalty disappears (maize 0.62).
- When November (Rabi) is chosen, lentil (0.38) overtakes the Kharif crop maize (0.37).

This is a rule layer evaluated by case study, not by accuracy. Measuring whether it improves yield would need multi-season field data, which is future work.

## N5: Explainable fit card and a learned crop catalogue
Each crop's P5–P95 envelope per feature is learned from the training data and stored in the `crops` table. The website shows, for every recommended crop, which measured value is inside, below or above its envelope. The SQL view `v_recommended_crops` uses the same envelopes as a transparent baseline that a clerk can verify by hand.

## Honest limitations (put these in the report)
- Moisture, soil temperature and soil type in the training data are **derived** from a public dataset with documented formulas; they are not field-measured.
- The Exp 3 sensor windows are simulated around dataset rows.
- The public dataset itself is widely used but its collection method is not documented in detail by its author.
- The VNBI weights and re-ranking factors are heuristic; they are not learned from yield data.
