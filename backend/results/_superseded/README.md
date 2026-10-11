# Superseded results

Historical runs kept for transparency. They are **not** the basis of the
reported numbers and should not be reused for new analyses.

| Folder | What it was | Superseded by |
|---|---|---|
| `Washington/llm_comparison/` | Generator comparison (Mistral vs. GPT-3.5 vs. Phi-3) as reported in the master's thesis. GPT-3.5 and Phi-3 received Mistral-formatted prompts, and the evaluated text was the output of the impersonation stage rather than the factual answer. | `Washington/llm_comparison_clean/` (model-appropriate prompts, factual answer evaluated directly) |
| `Washington/final_run_slm/` | Early exploratory run, three pipelines | `Washington/final_run_42Q/` |
| `Washington/smaller_chunks/` | Early exploratory run, three pipelines | `Washington/chunk_size_experiment/` |

The corrected protocol is documented in `backend/README_RERUN.md`.
