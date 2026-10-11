# Legacy scripts

Runner scripts used for earlier evaluation rounds, kept for reference.
For new runs use `backend/run_evaluation.py` (all pipelines, generators and
chunk sizes) and `backend/rerun_fixes.sh`.

| Script | Purpose | Replaced by |
|---|---|---|
| `run_all_gpu.py` | Original GPU sweep over the recursive pipelines | `run_evaluation.py` |
| `run_sent_sem_gpu.py` | GPU sweep over sentence/semantic pipelines | `run_evaluation.py` |
| `run_llm_comparison.py` | First generator comparison (superseded protocol) | `run_evaluation.py --llm ... --no-impersonation` |
| `prebuild_indices.py`, `prebuild_all_recursive.py` | Build BM25 indices up front | built automatically by `run_evaluation.py` |

The scripts use paths relative to `backend/` and import `packages`, so run
them as modules from `backend/`:

```bash
cd backend
python -m legacy.run_all_gpu
```
