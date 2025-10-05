# AI Workflows

BingXtg supports two AI stacks—HuggingFace transformers (preferred) and legacy spaCy models. This guide explains inference, training, testing, and configuration knobs.

## Inference Service

- `ai/inference/ai_service.py` bootstraps available models at import time.
- HuggingFace models are loaded when `HF_CLASSIFIER_MODEL_PATH` and `HF_NER_MODEL_PATH` are present; otherwise spaCy models (`CLASSIFIER_MODEL_PATH`, `NER_MODEL_PATH`, etc.) are used.
- **Model Precedence**: HuggingFace models take precedence over legacy spaCy models when both are available.
- The classifier predicts one of four classes:
  - `NON_SIGNAL`
  - `SIGNAL_LONG`
  - `SIGNAL_SHORT`
  - `SIGNAL_NONE`
- Token-level NER extracts entities such as trading pairs, entry price, targets, stop loss, leverage, and auxiliary metadata.

Run ad-hoc predictions with
```zsh
uv run python ai/inference/predict.py "Long BTCUSDT @ 62000, SL 59000, Targets 64000/66000"
```

## HuggingFace Training Pipeline

Location: `ai/training/hf`

1. **Export data**
   ```zsh
   uv run python ai/training/hf/export_data.py
   ```
   This creates `data_exports/classification_data.csv` and `data_exports/ner_data.jsonl`.

2. **Fine-tune models**
   Use the training scripts or Jupyter notebooks:
   - `train_classifier.py` / `train_classifier.ipynb` - Fine-tunes the signal classifier
   - `train_ner.py` / `train_ner.ipynb` - Fine-tunes the NER model

   > These scripts are GPU-friendly and can be run on Google Colab (T4 or better). Notebooks mirror the CLI scripts for interactive use.

3. **Evaluate models**
   ```zsh
   uv run python ai/training/hf/total_hf_ai_test.py
   ```

   The tests report precision/recall metrics across signal detection and entity extraction tasks.

4. **Deploy models**
   - Place the trained model directories under `ai/models/signal_classifier/` and `ai/models/ner_extractor/`.
   - Update `.env` with `HF_CLASSIFIER_MODEL_PATH` and `HF_NER_MODEL_PATH` if paths differ from defaults.
   - Restart running workers (`manage.py start_bot`, Django server) to load the new weights.

## Legacy spaCy Pipeline

Location: `ai/training`

- Use `classification_train.py`, `ner_training.py`, and `total_ai_test.py` for pure spaCy workflows.
- Models are smaller and CPU-friendly but generally less accurate than the transformer variants.
- This path is maintained for backward compatibility; prefer HuggingFace for new projects.

## Labeling Studio

- Invoke with `uv run python ai/labeling/main_textual.py`.
- Supports suggestions from multiple providers (Gemini, Groq, Anthropic, Cohere, Copilot) when API keys are set in `.env`.
- Outputs labeled samples directly into `total.db` for immediate reuse.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `MODEL_DIR` | Base directory for all models (default: `ai_model`). |
| `HF_CLASSIFIER_MODEL_PATH` | Path to HuggingFace classifier directory (default: `ai/models/signal_classifier`). |
| `HF_NER_MODEL_PATH` | Path to HuggingFace NER directory (default: `ai/models/ner_extractor`). |
| `IS_SIGNAL_MODEL_PATH` | Legacy spaCy signal detection model path (fallback). |
| `DIRECTION_MODEL_PATH` | Legacy spaCy direction classifier path (fallback). |
| `NER_MODEL_PATH` | Legacy spaCy NER model path (fallback). |
| `GOOGLE_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `GITHUB_TOKEN` | Optional LLM providers for assisted labeling. |

## Testing & Regression

- Always run `uv run python ai/training/hf/total_hf_ai_test.py` (or the spaCy equivalent) before shipping new models.
- Document metric deltas (precision, recall, F1) in release notes or pull requests.
