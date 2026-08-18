# Phase 2 Pilot: Physical Attribute Vocabulary (v2) - Result Record

## Experiment Setup

- Protocol: identical to config D (dynamic fusion OFF + z-score + class-balance sampling), only the text attribute vocabulary is replaced
- Vocabulary: `attributes/Indian_class_attributes_physical.json` (v2, 4 dimensions: spectral_reflectance / phenology / soil_background / canopy_structure)
- Baseline: natural vocabulary (v1, archived as `Indian_class_attributes_v1_natural.json`, same as config D)
- seeds = [42, 123, 456], epochs = 20, fixed last checkpoint, Indian Pines 6 unseen groups
- Output: `result_IP/phase2_physical_attrs_3seed_run.txt` (finished 2026-08-18 15:25, total 3179s)

## Per-Group OA Comparison (G1-G5 mean protocol, G6 single-class excluded)

| Group | Unseen classes | D (natural vocab) | Physical v2 | dOA |
|-------|----------------|-------------------|-------------|------|
| G1 | Corn-notill, Hay-windrowed, Woods | 86.54 +/- 6.47 | **51.83 +/- 2.34** | **-34.71** |
| G2 | Grass-trees, Buildings-G-T-D, Soybean-mintill | 90.77 +/- 2.42 | 89.49 +/- 2.34 | -1.28 |
| G3 | Grass-pasture-mowed, Stone-Steel-Towers, Corn-mintill | 86.86 +/- 6.96 | 85.81 +/- 3.07 | -1.05 |
| G4 | Corn, Alfalfa, Grass-pasture | 90.64 +/- 0.60 | **52.17 +/- 0.86** | **-38.47** |
| G5 | Wheat, Oats, Soybean-notill | 89.03 +/- 4.88 | 94.96 +/- 1.37 | +5.93 |
| G6 | Soybean-clean (excluded) | 100.00 | 100.00 | 0 |
| **G1-G5 mean** | - | **88.77 +/- 4.27** | **74.85 +/- 2.00** | **-13.92** |

Overall: OA 88.77 -> 74.85 (-13.92); AA 75.01 -> 61.70 (-13.31); Kappa 0.7253 -> 0.5357 (-0.19).

## Per-Class per-AA Comparison (3-seed mean, %)

| Class | D | Physical v2 | d |
|-------|-----|-------------|-----|
| Corn-notill | 99.9 | 81.8 | -18.1 |
| Hay-windrowed | 88.9 | 25.6 | **-63.3** |
| Woods | 40.6 | 31.8 | -8.8 |
| Grass-trees | 94.1 | 92.3 | -1.8 |
| Buildings-G-T-D | 27.4 | 15.4 | -12.0 |
| Soybean-mintill | 99.2 | 99.7 | +0.5 |
| Grass-pasture-mowed | 100.0 | 100.0 | 0.0 |
| Stone-Steel-Towers | 34.8 | 11.8 | **-23.0** |
| Corn-mintill | 92.4 | 93.9 | +1.5 |
| Corn | 99.0 | 99.6 | +0.6 |
| Alfalfa | 7.2 | 39.9 | **+32.7** |
| Grass-pasture | 94.5 | 30.1 | **-64.4** |
| Wheat | 56.3 | 99.7 | **+43.4** |
| Oats | 95.0 | 8.3 | **-86.7** |
| Soybean-notill | 95.8 | 95.8 | 0.0 |

## Honest Conclusion

1. **Physical vocab v2 is a net negative**: G1-G5 mean OA 74.85 vs 88.77, **-13.92**; AA and Kappa drop in sync.
2. Collapse concentrated in **G1 (-34.71)** and **G4 (-38.47)**; G5 slightly positive (+5.93), G2/G3 about flat.
3. Per-class effect is extremely polarized: a few classes improve a lot (Alfalfa +32.7, Wheat +43.4), most collapse (Oats -86.7, Grass-pasture -64.4, Hay-windrowed -63.3, Stone-Steel-Towers -23.0).
4. Mechanism: CLIP text encoder is trained on natural-image corpora; remote-sensing physical vocabulary is out-of-distribution. Physical prompts push class prototypes into unfamiliar CLIP-space regions, entangling several prototypes. The prototype-separability check predicted G1/G4/G5 better and G2/G3 worse, which did NOT match reality -> the pilot was necessary and the diagnostic cannot replace real training.
5. This follows the intended "cheap check -> pilot -> abandon if negative" workflow; recorded honestly, no data manipulation.

## Decision

- Rolled back `attributes/Indian_class_attributes.json` to the natural vocabulary (config-D state) so the repo default matches 88.77.
- The physical-vocabulary direction is **terminated**; no LongKou/Houston rollout.
- Vocabulary files kept as experiment archives: `Indian_class_attributes_physical_v1.json`, `Indian_class_attributes_physical.json`, `Indian_class_attributes_v1_natural.json`.

## Data File Index

- This run: `result_IP/phase2_physical_attrs_3seed_run.txt`
- Config-D run: `result_IP/ablation_D_multi_seed_run.txt`
- Prototype separability check: `result_IP/文本原型可分性校验.txt`
