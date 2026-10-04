# Pre-specified analysis plan for the revised (v2) evaluation

**Date fixed:** 4 October 2026. This document was committed before any augmented v2 classifier run existed. At the time of commit, the only v2 classifier result was the seed-42 baseline (`results/v2/siniflandirma/baseline_s42.json`); it is identical to the original baseline. The git history of this file is the record of the plan.

## Design

**Classifier.** ResNet-18 log-Mel classifier, trained with the same procedure as the original submission. Training runs for 30 epochs with a cosine schedule and **no early stopping**.

**Configurations.**

| Name | Synthetic training data |
|---|---|
| baseline | none |
| H1 | pretrained AudioLDM2/CLAP, English prompts |
| C2′ | fine-tuned CLAP, Cycle 1, English |
| C3′ | fine-tuned CLAP, Cycle 1, Turkish |
| C4′ | fine-tuned CLAP, Cycle 2, English |
| C4′-shift | content control: C4′ clips with labels cyclically shifted across the augmented classes |
| SpecAugment | none, masking only |
| SpecAugment + C4′ | C4′ clips, with masking |
| dose series | C4′ with 25 / 100 / 200 clips for Group A and the control |

All generative configurations add 50 clips to every augmented class, including the control.

**Classes.**
- Group A: dog, bird, weather.
- Group B: construction, religious place, farm/livestock. These are the three lowest baseline F1 values.
- Emergency service: reported separately, for continuity with the first target set.
- Control: other animals.

**Seeds.** baseline, H1 and C4′ use seeds 42–46; all other configurations use seeds 42–44. Each configuration is paired with the baseline trained with the same seed.

**CLAP.** Fine-tuned only on recordings from the classifier's training partition.

## Primary hypothesis (single, pre-specified)

> **P1.** Augmentation with C4′ synthetic clips increases the Group A macro-F1 relative to the baseline.

- **Endpoint:** the Group A macro-F1 difference, C4′ minus baseline. This is the mean of the class-F1 differences for dog, bird and weather, averaged over the five paired seeds.
- **Model and evaluation set:** the final-epoch model, evaluated on the validation and test partitions together (4,608 recordings). Because no early stopping is used, neither partition influences the final-epoch model.
- **Inference:** a cluster bootstrap over recordings, stratified by class (B = 10,000, generator seed 2026). In each replicate, both models are evaluated on the same resampled recordings. The test is two-sided at α = 0.05. We report the point estimate, the 95% percentile interval and the bootstrap p-value.
- **Implementation:** `python scripts/16_birincil_analiz.py --konfigler baseline C4p --degerlendirme son_valtest`.

## Secondary analyses

Same endpoint construction (final-epoch model, validation + test, paired seeds). Holm correction is applied across this family:

1. C4′ vs. baseline — Group B
2. C4′ vs. baseline — control class
3. H1 vs. baseline — Group A
4. C4′ vs. H1 — Group A (contribution of CLAP fine-tuning)
5. C2′ vs. baseline — Group A
6. C3′ vs. baseline — Group A
7. C3′ vs. C2′ — Group A (Turkish vs. English prompts)
8. C4′-shift vs. baseline — Group A (content control)
9. C4′ vs. C4′-shift — Group A (contribution of matched audio content)
10. SpecAugment vs. baseline — Group A
11. SpecAugment + C4′ vs. SpecAugment — Group A

**Also reported**, outside the family:
- the dose–response curve (25 / 50 / 100 / 200 clips; dose 50 = C4′);
- emergency service;
- 26-class macro-F1 (validation and test, mean ± SD over seeds);
- a logistic mixed model with a random recording slope, as a sensitivity analysis;
- the best-validation checkpoint evaluated on the test partition, segment-level McNemar tests, and the same analyses on the test partition only — all for continuity with the original submission.

## Rules

- Every run that is started is reported; nothing is excluded.
- An interrupted run is resumed from its saved state, which reproduces the uninterrupted run exactly (verified on 3–4 October 2026).
- Any deviation from this plan is reported in the manuscript and in the response letter, together with its reason.
