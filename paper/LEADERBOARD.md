# The LOAM paper's gLMBench results

Generated from `paper/records/` by `scripts/build_reference_table.py`; do not edit.
Last = the single-layer row (last layer); best = the layer-sweep row, with its tap index in
brackets. The best tap is chosen on the test split, so it is a retrospective upper bound.
Scores compare within a column only: never average AUROC, macro-F1 and Spearman.

| Model | Params | Essentiality AUROC, last | best (tap) | EC macro-F1, last | best (tap) | RNAGym Spearman | RNAGym scoring | Record |
|---|---:|---:|---:|---:|---:|---:|---|---|
| LOAM-25M | 25.5M | 0.7266 | 0.7266 (5) | 0.1815 | 0.1932 (4) | 0.1329 | sequence loglikelihood | [`glmb:ef7a2f14704c7258a9162958`](records/glmb_ef7a2f14704c7258a9162958.json) |
| LOAM-100M | 102.9M | 0.7574 | 0.7574 (8) | 0.3008 | 0.3008 (8) | 0.2027 | sequence loglikelihood | [`glmb:57ca8719fc72b4e44e70ccc3`](records/glmb_57ca8719fc72b4e44e70ccc3.json) |
| LOAM-340M | 340.0M | 0.7623 | 0.7722 (10) | 0.2776 | 0.3234 (10) | 0.2710 | sequence loglikelihood | [`glmb:7e22ae6e6afb5a856ff23dc2`](records/glmb_7e22ae6e6afb5a856ff23dc2.json) |
| LOAM-624M | 624.2M | 0.7703 | 0.7708 (15) | 0.2974 | 0.3939 (14) | 0.3176 | sequence loglikelihood | [`glmb:24879d8765197f4a0db44583`](records/glmb_24879d8765197f4a0db44583.json) |
| ProkBERT-mini | 20.6M | 0.7186 | 0.7250 (4) | 0.0865 | 0.0865 (6) | 0.1145 | masked marginal llr | [`glmb:35874e63ceb426b852f4cece`](records/glmb_35874e63ceb426b852f4cece.json) |
| ProkBERT-mini-c | 25.0M | 0.5565 | 0.5758 (1) | 0.0056 | 0.0056 (6) | 0.1296 | masked marginal llr | [`glmb:59dd1caa23d1b930e65627b5`](records/glmb_59dd1caa23d1b930e65627b5.json) |
| NTv3-100M | 106.5M | 0.6789 | 0.7053 (12) | 0.0465 | 0.1143 (14) | 0.1067 | masked marginal llr | [`glmb:ffc3488c7c0b5ca8bb6e7ed5`](records/glmb_ffc3488c7c0b5ca8bb6e7ed5.json) |
| GenomeOcean-100M | 119.6M | 0.7430 | 0.7611 (8) | 0.0799 | 0.1610 (7) | 0.1718 | sequence loglikelihood | [`glmb:a5a68d556cd33d19e3a28290`](records/glmb_a5a68d556cd33d19e3a28290.json) |
| gLM2-150M | 152.5M | 0.6799 | 0.6816 (28) | 0.0643 | 0.0643 (29) | 0.0652 | masked marginal llr | [`glmb:97f894dbf406d32617f34fdf`](records/glmb_97f894dbf406d32617f34fdf.json) |
| GenomeOcean-500M | 541.1M | 0.7441 | 0.7583 (10) | 0.0904 | 0.3448 (8) | 0.2479 | sequence loglikelihood | [`glmb:803cab8bb2aac55babc41b35`](records/glmb_803cab8bb2aac55babc41b35.json) |
| gLM2-650M | 670.6M | 0.6953 | 0.6953 (32) | 0.0595 | 0.0747 (10) | 0.0956 | masked marginal llr | [`glmb:f6c8e54295f55232d7502c3c`](records/glmb_f6c8e54295f55232d7502c3c.json) |
| Evo 1.5 (8k, 7B) | 6.45B | 0.7028 | 0.7779 (11) | 0.0234 | 0.3318 (7) | 0.3182 | sequence loglikelihood | [`glmb:1746872b80b27764a840cc61`](records/glmb_1746872b80b27764a840cc61.json) |
| Evo2-7B (residual stream) | 6.48B | 0.6992 | 0.7880 (5) | 0.0513 | 0.3721 (10) | 0.3421 | sequence loglikelihood | [`glmb:80643ec8dfc333d8344f7a21`](records/glmb_80643ec8dfc333d8344f7a21.json) |

Model-free k-mer floor (`kmer_floor.json`): bacbench-essentiality 0.6531 (k = 5); dgeb-ec-classification-dna 0.0191 (k = 4); rnagym-dms not applicable.
