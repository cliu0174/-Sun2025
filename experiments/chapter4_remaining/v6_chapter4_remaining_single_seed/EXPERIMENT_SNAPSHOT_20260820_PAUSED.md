# Chapter 4 remaining experiments — paused snapshot

Snapshot time: 2026-08-20 (Asia/Shanghai)

## Status

- The requested worker process (PID 27848) was already stopped when this snapshot was taken; no termination was required.
- No `Traceback`, `Error`, `Exception`, or `FAILED` entry was found in the latest error-log tail.
- The official redesigned protocol has **31 completed runs**: 5 label-budget runs (`100/50/30/20/10%`), 8 factorial-ablation runs at 10%, and 18 completed 2D-sensitivity runs at 10%.
- A legacy 5% label-budget result remains on disk for diagnostic history, but is **not** included in the official redesigned protocol or its completion count.

## Label-budget results (official; PI-MSCL, single seed)

| Label budget | MAE (%) | RMSE (%) | R2 | Result path |
|---:|---:|---:|---:|---|
| 100% | 1.5216 | 3.0135 | 0.8284 | `label_budget/pi_mscl_r1/split42_mask3262_train2262/result.json` |
| 50% | 2.7384 | 4.1133 | 0.6803 | `label_budget/pi_mscl_r0p5/split42_mask3262_train2262/result.json` |
| 30% | 2.3954 | 3.9485 | 0.7054 | `label_budget/pi_mscl_r0p3/split42_mask3262_train2262/result.json` |
| 20% | 2.5801 | 3.9396 | 0.7067 | `label_budget/pi_mscl_r0p2/split42_mask3262_train2262/result.json` |
| 10% | 2.9881 | 4.6501 | 0.5914 | `label_budget/pi_mscl_r0p1/split42_mask3262_train2262/result.json` |

Legacy diagnostic only (excluded): 5% — MAE 7.5288%, RMSE 10.1563%, R2 -0.9493 at `label_budget/pi_mscl_r0p05/split42_mask3262_train2262/result.json`.

## Factorial ablation results (10% labels)

| Variant | MAE (%) | RMSE (%) | R2 |
|---|---:|---:|---:|
| ms0_mono0_rate0 | 2.2105 | 3.2464 | 0.8008 |
| ms0_mono0_rate1 | 2.0581 | 3.1082 | 0.8174 |
| ms0_mono1_rate0 | 2.8396 | 4.0295 | 0.6932 |
| ms0_mono1_rate1 | 3.3366 | 4.9643 | 0.5343 |
| ms1_mono0_rate0 | 2.0727 | 3.5504 | 0.7618 |
| ms1_mono0_rate1 | 2.8431 | 4.3233 | 0.6468 |
| ms1_mono1_rate0 | 2.9881 | 4.6501 | 0.5914 |
| ms1_mono1_rate1 | 3.6300 | 5.0733 | 0.5136 |

All corresponding files are under `factorial_ablation_r0p1/<variant>/split42_mask3262_train2262/result.json`.

## Completed 2D sensitivity results (10% labels)

| lambda_mono | lambda_rate | MAE (%) | RMSE (%) | R2 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 2.0727 | 3.5504 | 0.7618 |
| 0 | 0.01 | 2.0993 | 3.1636 | 0.8109 |
| 0 | 0.05 | 2.2053 | 3.3420 | 0.7889 |
| 0 | 0.10 | 2.8431 | 4.3233 | 0.6468 |
| 0 | 0.20 | 2.5293 | 4.5611 | 0.6069 |
| 0 | 0.30 | 2.7328 | 4.0558 | 0.6891 |
| 0.05 | 0 | 2.0950 | 3.2278 | 0.8031 |
| 0.05 | 0.01 | 2.3553 | 3.6072 | 0.7541 |
| 0.05 | 0.05 | 2.0629 | 3.2909 | 0.7953 |
| 0.05 | 0.10 | 2.0210 | 3.0767 | 0.8211 |
| 0.05 | 0.20 | 2.9475 | 4.7780 | 0.5686 |
| 0.05 | 0.30 | 2.4738 | 3.3603 | 0.7866 |
| 0.10 | 0 | 2.8301 | 4.2205 | 0.6634 |
| 0.10 | 0.01 | 2.6783 | 3.7527 | 0.7339 |
| 0.10 | 0.05 | 3.3226 | 5.5848 | 0.4106 |
| 0.10 | 0.10 | 2.5728 | 3.6161 | 0.7529 |
| 0.10 | 0.20 | 2.3622 | 3.5635 | 0.7600 |
| 0.10 | 0.30 | 3.1500 | 4.7807 | 0.5681 |

All corresponding files are under `sensitivity_2d_r0p1/mono*_rate*/split42_mask3262_train2262/result.json`.

## Remaining work, intentionally deferred

The planned sensitivity grid is 6 values of lambda_mono (`0, 0.05, 0.1, 0.2, 0.3, 0.5`) by 6 values of lambda_rate (`0, 0.01, 0.05, 0.1, 0.2, 0.3`), hence 36 runs. Eighteen are complete; the remaining 18 runs are:

- lambda_mono = 0.2 with lambda_rate = 0, 0.01, 0.05, 0.1, 0.2, 0.3;
- lambda_mono = 0.3 with lambda_rate = 0, 0.01, 0.05, 0.1, 0.2, 0.3;
- lambda_mono = 0.5 with lambda_rate = 0, 0.01, 0.05, 0.1, 0.2, 0.3.

No partially running task was recorded as complete. Resume only the missing grid entries in a future execution.
