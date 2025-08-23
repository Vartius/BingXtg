import re
import json
from typing import List, Tuple, Dict, Any, Optional, Iterable
from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.nn as nn
import logging

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .database_manager import DatabaseManager

# Set up logging
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text preprocessing for message analysis."""

    def __init__(self):
        # Keep basic cleanup of URLs/emails; not used for parsing fields
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|"
            r"www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}|"
            r"[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\s|$)|"
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        )

    def clean_text(self, text: str) -> str:
        """
        Cleans and preprocesses text by removing URLs/emails and normalizing whitespace.
        Note: We preserve casing & punctuation (cased model benefits from them).
        """
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ============================ NER LABELS ============================
NER_LABELS: List[str] = [
    "O",
    "B-PAIR",
    "I-PAIR",
    "B-SL",
    "I-SL",
    "B-LEV",
    "I-LEV",
    "B-TGT",
    "I-TGT",
    "B-ENTRY",
    "I-ENTRY",
]
NER_LABEL2ID: Dict[str, int] = {lab: idx for idx, lab in enumerate(NER_LABELS)}
NER_ID2LABEL: Dict[int, str] = {idx: lab for lab, idx in NER_LABEL2ID.items()}

DIRECTION_LABEL2ID = {"none": 0, "long": 1, "short": 2}
DIRECTION_ID2LABEL = {v: k for k, v in DIRECTION_LABEL2ID.items()}


class MultiTaskDistilBert(DistilBertPreTrainedModel):
    """Single shared DistilBERT body with three heads: signal, direction, NER."""

    def __init__(
        self,
        config,
        num_labels_signal: int = 2,
        num_labels_direction: int = 3,
        num_ner_labels: int = len(NER_LABELS),
    ):
        super().__init__(config)
        self.num_labels_signal = num_labels_signal
        self.num_labels_direction = num_labels_direction
        self.num_ner_labels = num_ner_labels

        self.distilbert = DistilBertModel(config)
        hidden = config.dim
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Heads
        self.signal_head = nn.Linear(hidden, num_labels_signal)
        self.direction_head = nn.Linear(hidden, num_labels_direction)
        self.ner_head = nn.Linear(hidden, num_ner_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_signal=None,
        labels_direction=None,
        labels_ner=None,
        **kwargs,
    ):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # (B, T, H)
        # Mean pool over attention_mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            summed = (seq * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
        else:
            pooled = seq[:, 0]

        pooled = self.dropout(pooled)
        seq_d = self.dropout(seq)

        logits_signal = self.signal_head(pooled)
        logits_direction = self.direction_head(pooled)
        logits_ner = self.ner_head(seq_d)

        loss = None
        losses: Dict[str, torch.Tensor] = {}
        if labels_signal is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses["signal"] = loss_fct(
                logits_signal.view(-1, self.num_labels_signal), labels_signal.view(-1)
            )
        if labels_direction is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses["direction"] = loss_fct(
                logits_direction.view(-1, self.num_labels_direction),
                labels_direction.view(-1),
            )
        if labels_ner is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            losses["ner"] = loss_fct(
                logits_ner.view(-1, self.num_ner_labels), labels_ner.view(-1)
            )
        if losses:
            # Weight NER a bit higher (sequence-rich), adjust as needed
            loss = (
                1.0 * losses.get("signal", 0.0)
                + 1.0 * losses.get("direction", 0.0)
                + 1.5 * losses.get("ner", 0.0)
            )

        return {
            "loss": loss,
            "logits_signal": logits_signal,
            "logits_direction": logits_direction,
            "logits_ner": logits_ner,
        }


class MultiTaskTrainer(Trainer):
    """Custom Trainer that reads our three label keys and sums losses."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels_signal = inputs.pop("labels_signal", None)
        labels_direction = inputs.pop("labels_direction", None)
        labels_ner = inputs.pop("labels_ner", None)
        outputs = model(
            **inputs,
            labels_signal=labels_signal,
            labels_direction=labels_direction,
            labels_ner=labels_ner,
        )
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


@dataclass
class MTExample:
    text: str
    label_signal: int
    label_direction: int
    labels_ner: List[int]


class AIClassifier:
    """AI classifier with a DistilBERT backbone and multi-head outputs."""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        db_path: str = "messages.db",
        db_manager: Optional[DatabaseManager] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        # Allow injecting singleton DB manager to avoid extra connections
        self.db_manager = (
            db_manager if db_manager is not None else DatabaseManager(db_path)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AIClassifier using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[MultiTaskDistilBert] = None
        self.trainer: Optional[Trainer] = None
        # Configurable confidence threshold (env or ctor)
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            try:
                self.confidence_threshold = float(
                    os.getenv("AI_CONFIDENCE_THRESHOLD", "0.7")
                )
            except Exception:
                self.confidence_threshold = 0.7

    # ============================ DATA ACCESS ============================
    def _iter_training_rows(self) -> Iterable[Dict[str, Any]]:
        """Yield all labeled rows. Falls back gracefully if extended fields are missing.
        Includes non-signal samples so the signal head learns both classes and the
        direction/NER heads can learn 'none/O' when appropriate.
        """
        try:
            raw_rows: List[Any] = []
            # Prefer extended labeled data if available
            try:
                raw_rows = self.db_manager.get_extended_labeled_data()  # type: ignore[attr-defined]
            except Exception:
                raw_rows = (
                    self.db_manager.get_labeled_data()
                )  # Fallback (basic schema: message, is_signal)
            for rr in raw_rows:
                r: Dict[str, Any] = dict(rr)
                # Ensure keys exist for extended fields when training the multitask heads
                # so downstream code can .get(...) safely.
                if "direction" not in r:
                    r["direction"] = None
                # optional extended fields
                for k in ("pair", "stop_loss", "leverage", "targets", "entry"):
                    r.setdefault(k, None)
                yield r
        except Exception:
            logger.exception("Error fetching rows for training")
            return

    # ============================ LABEL BUILDERS ============================
    def _direction_to_id(self, val: Any) -> int:
        """Map DB/UI direction values to internal ids.
        DB stores: 0=LONG, 1=SHORT. Internal: 0=none, 1=long, 2=short.
        Also accepts strings like 'long'/'short' or 'buy'/'sell'.
        """
        if val is None:
            return DIRECTION_LABEL2ID["none"]
        if isinstance(val, (int, float)):
            iv = int(val)
            if iv == 0:
                return DIRECTION_LABEL2ID["long"]
            if iv == 1:
                return DIRECTION_LABEL2ID["short"]
            return DIRECTION_LABEL2ID["none"]
        s = str(val).strip().lower()
        if s in ("long", "buy", "0"):
            return DIRECTION_LABEL2ID["long"]
        if s in ("short", "sell", "1"):
            return DIRECTION_LABEL2ID["short"]
        return DIRECTION_LABEL2ID["none"]

    def _build_entity_spans(
        self, text: str, row: Dict[str, Any]
    ) -> List[Tuple[int, int, str]]:
        """Create weakly-supervised spans for NER from DB extended fields without regex.
        Returns a list of (start_char, end_char, label_name) where label_name in {PAIR, SL, LEV, TGT, ENTRY}.
        """
        spans: List[Tuple[int, int, str]] = []
        lower = text.lower()

        def add_spans_for_value(value_str: str, label: str):
            if not value_str:
                return
            v = value_str.strip()
            if not v:
                return
            start = 0
            lv = v.lower()
            while True:
                idx = lower.find(lv, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(v), label))
                start = idx + len(v)

        # pair
        pair = row.get("pair")
        if pair:
            # Also try normalized pair tokens such as BTC, BTCUSDT, BTC/USDT
            candidates = {str(pair).strip()}
            p = str(pair).strip()
            up = p.upper()
            candidates |= {
                up,
                up.replace("/", ""),
                up.replace("/", "") + "USDT",
                up + "USDT",
            }
            for c in candidates:
                add_spans_for_value(c, "PAIR")

        # stop_loss
        sl = row.get("stop_loss")
        if sl is not None:
            try:
                f = float(sl)
                cands = {f"{f}", f"{f:.2f}", f"{f:.3f}", f"{f:g}"}
                for c in sorted(cands, key=len, reverse=True):
                    add_spans_for_value(c, "SL")
            except Exception:
                pass

        # entry
        entry = row.get("entry")
        if entry is not None:
            try:
                f = float(entry)
                cands = {f"{f}", f"{f:.2f}", f"{f:.3f}", f"{f:g}"}
                for c in sorted(cands, key=len, reverse=True):
                    add_spans_for_value(c, "ENTRY")
            except Exception:
                pass

        # leverage (like 20x)
        lev = row.get("leverage")
        if lev is not None:
            try:
                f = float(lev)
                n = str(int(f)) if float(int(f)) == f else f"{f:g}"
                cands = {n, n + "x", n + " X", n + " x"}
                for c in cands:
                    add_spans_for_value(c, "LEV")
            except Exception:
                pass

        # targets: JSON string -> list
        targets = row.get("targets")
        if targets:
            arr: List[float] = []
            try:
                if isinstance(targets, str):
                    arr = json.loads(targets)
                elif isinstance(targets, (list, tuple)):
                    arr = list(targets)
            except Exception:
                arr = []
            for t in arr:
                try:
                    f = float(t)
                    cands = {f"{f}", f"{f:.2f}", f"{f:.3f}", f"{f:g}"}
                    for c in sorted(cands, key=len, reverse=True):
                        add_spans_for_value(c, "TGT")
                except Exception:
                    continue

        # Merge overlapping spans by preferring longer matches
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: List[Tuple[int, int, str]] = []
        for s in spans:
            if not merged or s[0] >= merged[-1][1]:
                merged.append(s)
            else:
                # overlap -> keep longer
                prev = merged[-1]
                if (s[1] - s[0]) > (prev[1] - prev[0]):
                    merged[-1] = s
        return merged

    def _align_ner_labels(
        self, text: str, tokens, offsets, spans: List[Tuple[int, int, str]]
    ) -> List[int]:
        labels = [NER_LABEL2ID["O"]] * len(tokens)
        # Mark special tokens as ignore later
        for start, end, t in spans:
            for i, (s, e) in enumerate(offsets):
                if s == e:  # special token
                    continue
                if s >= end or e <= start:
                    continue
                # overlaps
                tag_prefix = (
                    "B"
                    if s >= start and (i == 0 or offsets[i - 1][1] <= start)
                    else "I"
                )
                tag = f"{tag_prefix}-{t}"
                labels[i] = NER_LABEL2ID.get(tag, NER_LABEL2ID["O"])
        # Convert special tokens to -100
        for i, (s, e) in enumerate(offsets):
            if s == e:
                labels[i] = -100
        return labels

    # ============================ DATASET PREP ============================
    def _build_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        texts: List[str] = []
        labels_signal: List[int] = []
        labels_direction: List[int] = []
        ner_labels: List[List[int]] = []
        input_ids_list: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for r in rows:
            text = self.preprocessor.clean_text(r.get("message", ""))
            if not text:
                continue
            is_signal = int(r.get("is_signal") or 0)
            direction_id = self._direction_to_id(r.get("direction"))

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            tokens = enc["input_ids"]

            spans = self._build_entity_spans(text, r)
            labels_ner = self._align_ner_labels(text, tokens, offsets, spans)

            texts.append(text)
            labels_signal.append(is_signal)
            labels_direction.append(direction_id)
            ner_labels.append(labels_ner)
            input_ids_list.append(enc["input_ids"])
            attention_masks.append(enc["attention_mask"])

        ds = Dataset.from_dict(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_masks,
                "labels_signal": labels_signal,
                "labels_direction": labels_direction,
                "labels_ner": ner_labels,
            }
        )
        return ds

    # ============================ MODEL LIFECYCLE ============================
    def _build_model(self) -> MultiTaskDistilBert:
        model = MultiTaskDistilBert.from_pretrained(
            self.model_name,
            num_labels_signal=2,
            num_labels_direction=3,
            num_ner_labels=len(NER_LABELS),
        )
        model.to(self.device)  # type: ignore[arg-type]
        return model

    def _create_training_args(self, output_dir: str) -> TrainingArguments:
        """Create TrainingArguments with compatibility fallbacks for older transformers versions."""
        base_kwargs: Dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": 4,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "learning_rate": 3e-5,
        }
        # Try modern API first
        try:
            modern_kwargs: Dict[str, Any] = {
                **base_kwargs,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "logging_dir": f"{output_dir}/logs",
                "logging_steps": 25,
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "save_total_limit": 2,
                "fp16": torch.cuda.is_available(),
                "seed": 42,
                "dataloader_pin_memory": True,
                "dataloader_num_workers": 2,
                "report_to": [],
            }
            return TrainingArguments(**modern_kwargs)  # type: ignore[call-arg]
        except TypeError:
            logger.warning(
                "Falling back to minimal TrainingArguments (transformers version lacks some args)."
            )
            # Older API fallback (remove newer kwargs)
            try:
                older_kwargs: Dict[str, Any] = {
                    **base_kwargs,
                    "logging_dir": f"{output_dir}/logs",
                    "logging_steps": 25,
                    "warmup_steps": 0,
                    "weight_decay": 0.0,
                    "fp16": torch.cuda.is_available(),
                    "seed": 42,
                }
                return TrainingArguments(**older_kwargs)  # type: ignore[call-arg]
            except TypeError:
                # Ultra-minimal fallback
                try:
                    minimal_kwargs: Dict[str, Any] = {
                        "output_dir": output_dir,
                        "num_train_epochs": 4,
                        "per_device_train_batch_size": 8,
                        "learning_rate": 3e-5,
                    }
                    return TrainingArguments(**minimal_kwargs)  # type: ignore[call-arg]
                except TypeError:
                    # Last resort minimal args
                    return TrainingArguments(output_dir=output_dir)  # type: ignore[call-arg]

    def train_model(self, output_dir: str = "./ai_model") -> bool:
        """Train multi-task model on filtered data only."""
        try:
            rows = list(self._iter_training_rows())
            if len(rows) < 20:
                logger.warning(
                    f"Training skipped: need at least 20 samples after filtering, got {len(rows)}"
                )
                return False

            # Split (stratify on signal to keep class balance if possible)
            y = [int(r.get("is_signal") or 0) for r in rows]
            stratify = y if len(set(y)) > 1 else None
            train_rows, val_rows = train_test_split(
                rows, test_size=0.2, random_state=42, stratify=stratify
            )
            train_ds = self._build_dataset(train_rows)
            val_ds = self._build_dataset(val_rows)

            # Warm-start from existing model dir if present
            model_dir = Path(output_dir)
            if (model_dir / "config.json").exists():
                logger.info(f"Continuing training from existing model at {output_dir}")
                model = MultiTaskDistilBert.from_pretrained(
                    output_dir,
                    num_labels_signal=2,
                    num_labels_direction=3,
                    num_ner_labels=len(NER_LABELS),
                )
                model.to(self.device)  # type: ignore[arg-type]
                self.model = model
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
                except Exception:
                    logger.warning(
                        "Failed to load tokenizer from existing model dir; using base tokenizer."
                    )
            else:
                self.model = self._build_model()

            training_args = self._create_training_args(output_dir)

            def data_collator(
                features: List[Dict[str, Any]],
            ) -> Dict[str, torch.Tensor]:
                # Pad input_ids & attention_mask
                batch_input_ids = [
                    torch.tensor(f["input_ids"], dtype=torch.long) for f in features
                ]
                batch_attention = [
                    torch.tensor(f["attention_mask"], dtype=torch.long)
                    for f in features
                ]
                batch_labels_signal = torch.tensor(
                    [f["labels_signal"] for f in features], dtype=torch.long
                )
                batch_labels_direction = torch.tensor(
                    [f["labels_direction"] for f in features], dtype=torch.long
                )
                # Pad ner labels
                max_len = max(x.size(0) for x in batch_input_ids)
                padded_ids = torch.stack(
                    [
                        nn.functional.pad(x, (0, max_len - x.size(0)), value=0)
                        for x in batch_input_ids
                    ]
                )
                padded_att = torch.stack(
                    [
                        nn.functional.pad(x, (0, max_len - x.size(0)), value=0)
                        for x in batch_attention
                    ]
                )
                padded_ner = []
                for f in features:
                    ner_lab = f["labels_ner"]
                    pad_len = max_len - len(ner_lab)
                    padded_ner.append(
                        torch.tensor(ner_lab + [-100] * pad_len, dtype=torch.long)
                    )
                padded_ner = torch.stack(padded_ner)
                return {
                    "input_ids": padded_ids,
                    "attention_mask": padded_att,
                    "labels_signal": batch_labels_signal,
                    "labels_direction": batch_labels_direction,
                    "labels_ner": padded_ner,
                }

            self.trainer = MultiTaskTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=data_collator,
            )

            self.trainer.train()
            # Save
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Training completed and saved to {output_dir}")
            return True
        except Exception:
            logger.exception("Training failed")
            return False

    def load_model(self, model_path: str = "./ai_model") -> bool:
        try:
            self.model = MultiTaskDistilBert.from_pretrained(
                model_path,
                num_labels_signal=2,
                num_labels_direction=3,
                num_ner_labels=len(NER_LABELS),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)  # type: ignore[arg-type]
            logger.info(f"Model loaded from {model_path} onto {self.device}")
            return True
        except Exception:
            logger.exception(f"Error loading model from {model_path}")
            return False

    # ============================ INFERENCE ============================
    def predict(self, text: str) -> Tuple[int, float]:
        """Predict binary signal label and confidence from the signal head."""
        if not self.model:
            logger.warning("Predict called before model is loaded.")
            return 0, 0.0
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return 0, 0.0
        enc = self.tokenizer(
            clean_text, return_tensors="pt", truncation=True, max_length=256
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.eval()
        with torch.inference_mode():
            out = self.model(**enc)
            logits = out["logits_signal"]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            pred_i = int(pred.item())
            conf_f = float(conf.item())
            # Log prediction result (truncate text for readability)
            prev = clean_text[:120].replace("\n", " ")
            logger.info(
                f"AI Predict: label={pred_i} conf={conf_f:.3f} text='{prev}...'"
            )
            return pred_i, conf_f

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """Vectorized predictions. Returns (label, confidence) per text."""
        if not self.model or not texts:
            return [(0, 0.0) for _ in texts]
        cleaned = [self.preprocessor.clean_text(t or "") for t in texts]
        non_empty = [(i, t) for i, t in enumerate(cleaned) if t]
        results: List[Tuple[int, float]] = [(0, 0.0)] * len(texts)
        if not non_empty:
            return results
        enc = self.tokenizer(
            [t for _, t in non_empty],
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.eval()
        with torch.inference_mode():
            out = self.model(**enc)
            probs = torch.softmax(out["logits_signal"], dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            for (orig_idx, _), p, c in zip(non_empty, pred.tolist(), conf.tolist()):
                results[orig_idx] = (int(p), float(c))
        return results

    def _decode_ner(
        self, text: str, enc: Dict[str, torch.Tensor], logits_ner: torch.Tensor
    ) -> Dict[str, Any]:
        ids = enc["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        probs = torch.softmax(logits_ner, dim=-1)[0]
        labels = torch.argmax(probs, dim=-1).tolist()
        # Reconstruct entities
        entities: Dict[str, List[str]] = {
            k: [] for k in ["PAIR", "SL", "LEV", "TGT", "ENTRY"]
        }
        current_type = None
        current_tokens: List[str] = []
        pad_tok = getattr(self.tokenizer, "pad_token", None)
        for tok, lid in zip(tokens, labels):
            if tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, pad_tok):
                # finalize current
                if current_type and current_tokens:
                    entities[current_type].append(
                        self.tokenizer.convert_tokens_to_string(current_tokens)
                    )
                current_type, current_tokens = None, []
                continue
            tag = NER_ID2LABEL.get(lid, "O")
            if tag == "O":
                if current_type and current_tokens:
                    entities[current_type].append(
                        self.tokenizer.convert_tokens_to_string(current_tokens)
                    )
                current_type, current_tokens = None, []
            else:
                prefix, etype = tag.split("-", 1)
                if prefix == "B":
                    if current_type and current_tokens:
                        entities[current_type].append(
                            self.tokenizer.convert_tokens_to_string(current_tokens)
                        )
                    current_type, current_tokens = etype, [tok]
                else:  # I-
                    if current_type == etype:
                        current_tokens.append(tok)
                    else:
                        # Start new span
                        if current_type and current_tokens:
                            entities[current_type].append(
                                self.tokenizer.convert_tokens_to_string(current_tokens)
                            )
                        current_type, current_tokens = etype, [tok]
        if current_type and current_tokens:
            entities[current_type].append(
                self.tokenizer.convert_tokens_to_string(current_tokens)
            )

        # Post-process values
        def pick_first(lst: List[str]) -> Optional[str]:
            return lst[0].strip() if lst else None

        def to_float_list(lst: List[str]) -> List[float]:
            out: List[float] = []
            for s in lst:
                ss = s.replace(" ", "").replace(",", ".")
                try:
                    # Drop trailing x for leverage targets/values
                    if ss.lower().endswith("x"):
                        ss = ss[:-1]
                    out.append(float(ss))
                except Exception:
                    continue
            return out

        def to_single_float(s: Optional[str]) -> Optional[float]:
            if not s:
                return None
            vals = to_float_list([s])
            return vals[0] if vals else None

        pair = pick_first(entities["PAIR"])
        stop_loss = pick_first(entities["SL"])
        entry = pick_first(entities["ENTRY"])
        leverage_vals = to_float_list(entities["LEV"])
        target_vals = to_float_list(entities["TGT"])

        result: Dict[str, Any] = {
            "pair": pair.upper() if pair else None,
            "stop_loss": to_single_float(stop_loss),
            "entry": to_single_float(entry),
            "leverage": leverage_vals[0] if leverage_vals else None,
            "targets": json.dumps(target_vals) if target_vals else None,
        }
        return result

    def extract_signal_fields(self, text: str) -> Dict[str, Any]:
        """Run the model and extract structured fields using the NER and direction heads (no regex)."""
        if not self.model:
            logger.warning("extract_signal_fields called before model is loaded.")
            return {}
        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return {}
        enc = self.tokenizer(
            clean_text, return_tensors="pt", truncation=True, max_length=256
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.eval()
        with torch.no_grad():
            out = self.model(**enc)
            # Decode NER fields
            fields = self._decode_ner(clean_text, enc, out["logits_ner"].cpu())
            # Decode direction
            dir_logits = out["logits_direction"]
            dir_pred = int(
                torch.argmax(torch.softmax(dir_logits, dim=-1), dim=-1).item()
            )
            # Map internal ids (0 none, 1 long, 2 short) -> UI values (0 LONG, 1 SHORT) or None
            ui_direction: Optional[int]
            if dir_pred == 1:
                ui_direction = 0  # LONG
            elif dir_pred == 2:
                ui_direction = 1  # SHORT
            else:
                ui_direction = None
            fields["direction"] = ui_direction
            # Log extracted info
            preview = clean_text[:200].replace("\n", " ")
            logger.info(
                f"AI Extract: direction={ui_direction} fields={fields} text='{preview}'"
            )
            return fields

    # ============================ ACTIVE LEARNING ============================
    def get_uncertain_samples(
        self, texts: List[str], n_samples: int
    ) -> List[Tuple[int, str, float]]:
        if not self.model or not texts:
            return []
        result: List[Tuple[int, str, float]] = []
        for i, (_pred, conf) in enumerate(self.predict_batch(texts)):
            if conf < self.confidence_threshold:
                result.append((i, texts[i], conf))
        result.sort(key=lambda x: x[2])
        return result[:n_samples]


class ActiveLearningManager:
    """Manages the active learning workflow by suggesting messages for labeling."""

    def __init__(self, classifier: AIClassifier, db_manager: DatabaseManager):
        self.classifier = classifier
        self.db_manager = db_manager

    def suggest_next_messages(self, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Requesting {n_suggestions} suggestions for active learning.")
        try:
            unlabeled_messages = self.db_manager.get_unlabeled_messages(limit=200)
            if not unlabeled_messages:
                return []
            message_texts = [msg["message"] for msg in unlabeled_messages]

            uncertain_results: List[Tuple[int, str, float]] = []
            if self.classifier.model:
                for i, (_pred, confidence) in enumerate(
                    self.classifier.predict_batch(message_texts)
                ):
                    if confidence < self.classifier.confidence_threshold:
                        uncertain_results.append((i, message_texts[i], confidence))
                uncertain_results.sort(key=lambda x: x[2])
                uncertain_results = uncertain_results[:n_suggestions]
            else:
                import random

                idxs = random.sample(
                    range(len(message_texts)), min(n_suggestions, len(message_texts))
                )
                uncertain_results = [(i, message_texts[i], 0.0) for i in idxs]

            suggestions = []
            for idx, _t, conf in uncertain_results:
                s = dict(unlabeled_messages[idx])
                s["confidence"] = conf
                suggestions.append(s)
            return suggestions
        except Exception:
            logger.exception("Failed to suggest next messages for labeling.")
            return []

    def get_training_recommendations(self) -> Dict[str, Any]:
        try:
            return self.db_manager.get_training_stats()
        except Exception:
            logger.exception("Failed to get training recommendations.")
            return {}

