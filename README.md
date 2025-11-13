# Soft Token Interpolation for Masked dLLM (LLaDA)

This repository contains a training-free extension to masked diffusion LLMs that interpolates between a hard mask and a decoded token. Instead of collapsing masked positions to a single `[MASK]` embedding, we retain the model's probility distribution from the previous step, mix candidate embeddings, and feed the resulting **Soft Token** back into the next denoising pass. The outcome is a diffusion trajectory that keeps semantic information alive even when a position remains masked, thus improving information transition flow between diffusion steps.

The codebase originates from the llada part in [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) implementation, and the idea of mixing candidate embeddings is inspired by the latent reasoning method proposed in [Soft Think](https://arxiv.org/abs/2505.15778).

---

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Soft Token Sampling Pipeline](#soft-token-sampling-pipeline)
5. [Running the Provided Experiments](#running-the-provided-experiments)
6. [Plotting and Reporting](#plotting-and-reporting)
7. [Extending the Soft Token Mechanism](#soft-token-parameters)

---

## Conceptual Overview

- **Problem.** In masked diffusion decoding, only unmasked positions contribute signal during a denoising step. Specifically, early steps therefore operate on a sea of `[MASK]` embeddings, discarding useful information gathered in previous iterations.
- **Idea.** Keep a continuous latent between "mask" and "token". When a position remains masked, store the predicted token distribution from the previous step and form a soft embedding by mixing candidate token embeddings ($\mathbf{e}_{\text{soft token}} = \sum_i p_i \cdot \mathbf{e}_{\text{candidate token}_i}$).
- **Benefits.**
  - Information from previous steps flows through the model even if the discrete token has not been committed.
  - Sampling is more efficient, because fewer steps are wasted rediscovering what was already inferred.
  - This approach requires no retraining and introduces only minimal computational overhead.

---


## Soft Token Sampling Pipeline

### 1. Candidate Selection (`generate.py`)

1. Decode with the base samplers until logits for the current block are available.
2. Call `get_transfer_index_soft_token` to:
   - Decide which masked positions transition this step.
   - Record the top-`k` (default `k=5`) token ids with probabilities for **all** positions.
   - Append a synthetic `[MASK]` candidate with probability `1 − p_max + bias` (`bias` defaults to `0.5`).
3. Update with the committed tokens and cache the `(soft_token, prob)` tensors.

References: `generate_with_dual_cache_soft_token` and `get_transfer_index_soft_token` in `generate.py`.

### 2. Soft Embedding Injection (`model/modeling_llada.py`)

Modify the model forward to accept two optional tensors: `soft_token` (top-`k` ids per position) and `prob` (their weights). When supplied:

1. Retrieve the base token embeddings $\mathbf{e}_{\text{candidate token}_i}$ for all candidate tokens (via `F.embedding`).
2. Compute the weighted mixture to form the soft embedding
   $\mathbf{e}_{\text{soft token}} = \sum_i p_i \cdot \mathbf{e}_{\text{candidate token}_i},$
   producing $\mathbf{e}_{\text{soft token}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$.
3. For positions that remain `[MASK]`, replace their embeddings with $\mathbf{e}_{\text{soft token}}$; decoded (unmasked) positions retain their original embeddings.

References: `LLaDAModelLM.forward` in `model/modeling_llada.py`.

---

## Running the Provided Experiments

1. **Choose a checkpoint.**  
   - Base: `GSAI-ML/LLaDA-8B-Base`  
   - Instruct: `GSAI-ML/LLaDA-8B-Instruct`
2. **Set the environment.**
   ```bash
   export HF_ALLOW_CODE_EVAL=1
   export HF_DATASETS_TRUST_REMOTE_CODE=true
   ```
3. **Launch a sweep.**
   ```bash
   cd test/Soft-token-llada
   bash eval_dual_soft_token_base.sh   # or eval_dual_soft_token_inst.sh
   ```
   The script will:
   - Detect the available GPUs and pass them to `accelerate launch`.
   - Loop over `task_list=(mbpp gsm8k)` and a list of confidence thresholds.
   - For each combination, run both the soft-token variant (outputs under `eval_results_soft_token/...`) and the baseline dual-cache decoder (outputs under `eval_results/...`).
4. **Inspect outputs.**
   - `results_{timestamp}.json` contains the `lm-eval-harness` metrics.
   - `summary.txt` logs tokens/sec and total NFEs.

---

## Plotting and Reporting

- Use `plotting/extract.py` to aggregate gsm8k runs into a single JSON file keyed by threshold. It reads both accuracy metrics and the `summary.txt` NFEs:
  ```bash
  python plotting/extract.py eval_results_soft_token/base_parallel_dual gsm8k_soft_vs_base.json --task gsm8k --method-key soft-token
  ```
- Use `plotting/plot_gsm8k.py` to create comparison figures. Update the paths/method keys before running.

- **Example results.** Soft tokens consistently outperform the baseline on the accuracy–latency frontier for GSM8K using both LLaDA-8B-Instruct and LLaDA-8B-Base with different confidence threshold.

  ![gsm8k accuracy vs latency for LLaDA-8B-Instruct](plotting/gsm8k_llada_inst_baseline_vs_soft.png)

  ![gsm8k accuracy vs latency for LLaDA-8B-Base](plotting/gsm8k_llada_base_baseline_vs_soft.png)

---

## Soft Token Parameters

- **Vary `k_soft`.** Adjust the number of candidate tokens `k_soft` in `get_transfer_index_soft_token` for expressiveness.
- **Vary `addition_prob_mask`.** Change mask probility `bias` in `get_transfer_index_soft_token` to set how aggressively the model can revert to `[MASK]`.
