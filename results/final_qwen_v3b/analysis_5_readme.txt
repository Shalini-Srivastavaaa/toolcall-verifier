TABLE (363 real rows): AUROC (CI) overall | natural | recall@5%FA | recall@10%FA | recall at 0.5 / false-flag at 0.5 | natural recall@5% / @10%
| ModernBERT v3b           | 0.771 (0.720-0.818) | 0.753 (0.699-0.807) | induced 0.724 (0.606-0.832) | 0.229 | 0.329 | 0.494 / 0.197 | 0.133 / 0.286 |
| Qwen2.5-1.5B             | 0.765 (0.714-0.814) | 0.803 (0.753-0.853) | induced 0.533 (0.370-0.691) | 0.106 | 0.218 | 0.429 / 0.187 | 0.204 / 0.276 |
| ensemble: average        | 0.780 (0.730-0.825) | 0.785 (0.733-0.838) | induced 0.676 (0.533-0.806) | 0.124 | 0.312 | 0.465 / 0.161 | 0.133 / 0.337 |
| ensemble: rank-average   | 0.786 (0.736-0.831) | 0.791 (0.740-0.842) | induced 0.674 (0.528-0.805) | 0.147 | 0.282 | 0.782 / 0.358 | 0.153 / 0.276 |
| GPT-5-mini               | 0.719 (0.666-0.770) | 0.695 (0.620-0.760) | induced 0.698 (0.567-0.820) | 0.100 | 0.471 | 0.547 / 0.145 | 0.245 / 0.510 |
| GPT-5-nano               | 0.619 (0.562-0.677) | 0.623 (0.549-0.691) | induced 0.586 (0.449-0.728) | 0.112 | 0.229 | 0.435 / 0.218 | 0.143 / 0.265 |
| length only              | 0.675 (0.620-0.730) | 0.626 (0.556-0.693) | induced 0.559 (0.424-0.700) | 0.212 | 0.329 | 0.094 / 0.021 | 0.204 / 0.265 |
| tool name only           | 0.716 (0.661-0.767) | 0.705 (0.643-0.765) | induced 0.579 (0.440-0.718) | 0.076 | 0.135 | 0.335 / 0.155 | 0.051 / 0.071 |

PER CLASS recall at 0.5 (errors of the class; n total / n natural), ModernBERT | Qwen | ensemble (average) | GPT-5-mini | GPT-5-nano
| 1 wrong arg              | 30 / 21 | 0.43 | 0.27 | 0.33 | 0.77 | 0.53 |
| 2 stale state            | 4 / 0 | 1.00 | 0.75 | 1.00 | 1.00 | 0.75 |
| 3 wrong tool             | 18 / 14 | 0.44 | 0.39 | 0.28 | 0.67 | 0.67 |
| 4 repeated call          | 19 / 16 | 0.63 | 0.79 | 0.84 | 0.63 | 0.37 |
| 5 failure to propagate   | 20 / 1 | 0.40 | 0.50 | 0.35 | 0.55 | 0.50 |
| 6 ignored error          | 16 / 1 | 0.50 | 0.31 | 0.25 | 0.50 | 0.25 |
| 7 policy violation       | 63 / 45 | 0.49 | 0.40 | 0.52 | 0.37 | 0.35 |

false-flag rate on the 193 correct rows at 0.5: ModernBERT v3b 0.20, Qwen2.5-1.5B 0.19, ensemble: average 0.16, GPT-5-mini 0.15, GPT-5-nano 0.22

SPEED AND COST (speed_cost.json): {
"modernbert_cpu": {
"n": 363,
"device": "arm CPU, torch 2.15.0, 4 threads, fp32, batch of 1",
"total_p50_s": 0.6793136669998603,
"total_p95_s": 2.2877790455001104,
"total_mean_s": 0.8648657027300242,
"total_max_s": 2.5860352079998847,
"forward_p50_s": 0.6764897079999628,
"forward_p95_s": 2.2758625786000266,
"format_plus_tokenize_mean_s": 0.004454510920108248,
"tokens_p50": 1863.0,
"tokens_p95": 
