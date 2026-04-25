# Private ASME Validation

ANVIL's public benchmark uses SPES-1 and public-domain engineering documents.
Licensed ASME standards can be used only in a private local validation path.

## Rules

- Do not commit ASME PDFs, screenshots, OCR output, parser output, indexes, prompts, raw responses, retrieved chunks, or agent transcripts.
- Keep licensed inputs under `data/private/`.
- Keep private run artifacts under `data/private_runs/`.
- Publish only sanitized aggregate metrics: dataset size, pass rate, metric aggregates, edition label, and date.
- Do not publish ASME quoted text or paragraph excerpts without permission.

ASME states that its standards are copyrighted and that republishing excerpts
requires permission. The public repo therefore remains reproducible with
SPES-1, while ASME validation is reproducible only by someone with a licensed
copy.

## Suggested Local Layout

```text
data/private/asme/
  asme_viii_private.md
  asme_eval_v1.json

data/private_runs/
  <run_id>/
```

## Private Eval Command

```bash
uv run anvil eval \
  --backend fake \
  --dataset data/private/asme/asme_eval_v1.json \
  --standard data/private/asme/asme_viii_private.md \
  --output-root data/private_runs \
  --dataset-version asme-private-v1 \
  --min-pass-rate 0.0
```

Before sharing the repo, run:

```bash
uv run python scripts/audit_private_artifacts.py
```
