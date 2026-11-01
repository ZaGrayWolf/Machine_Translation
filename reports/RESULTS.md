# ICT4403 Assignment 3 — Extracted Results

> This package was reconstructed from the **saved outputs embedded in the uploaded Jupyter notebook**. It contains the figures and textual/table outputs visible in that notebook. The CSV files in `tables/` are transcriptions of the displayed notebook tables, not the original Kaggle CSV artifacts. It does not contain the original Kaggle working directory, translation CSVs, full raw dataset, or model checkpoints.

## Executive summary

- All four systems are marked evaluated in the notebook: Basic Transformer, mT5, LLaMA and Gemma.
- The primary quality comparison uses the same **500 test sentences** and identical source/reference sequences for every system. Primary decoding is greedy.
- **Gemma** records the highest primary BLEU (**15.80**) and chrF (**41.40**). Its advantage over LLaMA, the runner-up, is reported as statistically significant after Holm correction for both metrics using paired bootstrap resampling (300 resamples).
- **Basic Transformer** has the lowest batched latency per sentence (**66.47 ms**) and highest throughput (**15.04 sentences/s**).
- Systems differ in pretraining, parameter count, tokenizer and adaptation method; this is a comparison of these four systems under this run configuration, not a controlled architecture-only comparison.

## Dataset and run configuration

- Dataset: IIT Bombay English–Hindi Parallel Corpus; source: [Hugging Face dataset card](https://huggingface.co/datasets/cfilt/iitb-english-hindi). The notebook records a CC BY-NC 4.0 dataset-card license and notes non-commercial research use.
- Run profile: `session`; model preset: `plan`; seed: `42`; all four models enabled; `RESUME=True`.
- Final selected splits: 20,000 training pairs, 520 validation pairs, 500 test pairs. The notebook reports the test set is shared across all four systems.
- Train split mean length: English 15.75 words, Hindi 16.97 words. Validation: 17.71 / 17.90. Test: 19.43 / 22.49. See `tables/dataset_split_statistics.csv` and plot 01.
- Bootstrap repetitions: 300. SentencePiece vocabulary: 8,000.

## Primary translation metrics

| Model | BLEU (95% CI) | chrF (95% CI) | Length ratio | Empty rate | Off-target script | Number preservation |
|---|---:|---:|---:|---:|---:|---:|
| Basic Transformer | 2.09 (1.64–2.50) | 18.38 (17.80–18.95) | 1.01 | 0.00 | 0.00 | 0.12 |
| mT5 | 0.68 (0.51–0.87) | 12.31 (11.80–12.82) | 1.86 | 0.00 | 0.00 | 0.14 |
| LLaMA | 10.32 (9.18–11.62) | 34.34 (33.09–35.41) | 1.07 | 0.00 | 0.00 | 0.89 |
| Gemma | **15.80 (14.72–16.92)** | **41.40 (40.34–42.43)** | 0.96 | 0.00 | 0.00 | 0.97 |

The notebook's significance output prints p-values as `0.0`; interpret these as values rounded to the displayed precision, not as proof that the underlying p-value is mathematically exactly zero. Full pairwise output is preserved in `reports/extracted_notebook_outputs.txt`.

Metric configuration recorded in the notebook: SacreBLEU 2.6.0; BLEU with mixed case, no effective order, exponential smoothing, and Hindi tokenization after pre-tokenization; chrF with character n-gram order 6, word order 0, beta 2, no whitespace, NFC text.

## Efficiency and adaptation

| Model | Adaptation | Training time (s) | Total params | Trainable params | Batched ms/sentence | Batch-1 mean ms | Throughput (sent/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Basic Transformer | From scratch | 542.92 | 33,523,712 | 33,523,712 | **66.47** | 189.99 | **15.04** |
| mT5 | Full fine-tuning, `google/mt5-small` | 825.65 | 300,176,768 | 300,176,768 | 645.54 | 1,753.29 | 1.55 |
| LLaMA | LoRA r=16, alpha=32, `meta-llama/Llama-3.2-1B` | 4,998.27 | 1,247,086,592 | 11,272,192 | 1,009.24 | 2,674.61 | 0.99 |
| Gemma | LoRA r=16, alpha=32, `google/gemma-3-1b-pt` | 7,765.92 | 1,012,931,712 | 13,045,760 | 630.70 | 2,934.07 | 1.59 |

The training time includes the timing measure written by the notebook; treat comparisons as run-specific. The Gemma run cell says it reused existing results (`RESUME=True`), while the final comparison table records its training time and metrics from the loaded summary.

## Temperature experiment

Temperature analysis is supplementary and uses the notebook's analysis subset, not the primary 500-sentence test comparison. Each row reports `n_samples=1`; do not treat the numbers as estimates from multiple sampling repetitions. At temperature 1.5, the recorded off-target script rate is 0.958 for LLaMA, alongside a sharp fall in its BLEU/chrF. The complete table is in `tables/temperature_sampling_analysis.csv`; plot 09 visualises these outputs.

## Qualitative and ethics outputs

The notebook saved a 21-example qualitative comparison covering strong examples, possible omissions/additions, number preservation, off-target script/copy/empty outputs, model disagreement, and a stratified sample. It also reports automated gender-marking, formality, religion minimal-pair, number/name/negation and sensitive-content probes. See the full saved text in `reports/extracted_notebook_outputs.txt`. The notebook explicitly says human evaluation was **not completed**; the blinded rubric template is not human-rating evidence.

## Figure index

1. `01_dataset_length_distributions.png` — English and Hindi sentence-length distributions over the selected splits.
2. `02_training_loss_curves.png` — training loss and validation NLL by model; values are not directly comparable across tokenizers/models.
3. `03_training_time.png` — training/fine-tuning wall-clock time.
4. `04_bleu_scores_bootstrap_ci.png` — BLEU with paired-bootstrap intervals.
5. `05_chrf_scores_bootstrap_ci.png` — chrF with paired-bootstrap intervals.
6. `06_inference_latency_throughput.png` — batch-one latency and batched throughput.
7. `07_parameter_counts.png` — total and trainable parameters (log scale).
8. `08_model_comparison_table.png` — consolidated comparison table as embedded in the notebook.
9. `09_temperature_analysis.png` — supplementary sampling-temperature experiment.

## Important recovery note

The ZIP that the notebook printed (`/kaggle/working/assignment3_outputs.zip`, 1.1 MB) was not attached to the `.ipynb`. This archive contains the nine figures and extracted text/tables that were actually embedded in the notebook file. It does **not** recreate the missing original ZIP or the underlying translation/checkpoint files.
