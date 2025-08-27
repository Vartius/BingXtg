import re
import json
from typing import List, Tuple, Dict, Any, Optional, Iterable
import os
from pathlib import Path

import torch
import torch.nn as nn
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    DebertaV2Tokenizer,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .database_manager import DatabaseManager

# Set up logging
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text preprocessing for message analysis."""

    def __init__(self):
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|"
            r"www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}|"
            r"[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\s|$)|"
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        )

    def clean_text(self, text: str) -> str:
        """Cleans and preprocesses text by removing URLs/emails and normalizing whitespace."""
        if not isinstance(text, str):
            return ""
        text = self.url_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ============================ LABEL MAPPINGS ============================
NER_LABELS: List[str] = [
    "O",
    "B-PAIR",
    "I-PAIR",
    "B-STOP_LOSS",
    "I-STOP_LOSS",
    "B-LEVERAGE",
    "I-LEVERAGE",
    "B-TAKE_PROFIT",
    "I-TAKE_PROFIT",
    "B-ENTRY",
    "I-ENTRY",
]
NER_LABEL2ID: Dict[str, int] = {lab: idx for idx, lab in enumerate(NER_LABELS)}
NER_ID2LABEL: Dict[int, str] = {idx: lab for lab, idx in NER_LABEL2ID.items()}

DIRECTION_LABEL2ID = {"none": 0, "long": 1, "short": 2}
DIRECTION_ID2LABEL = {v: k for k, v in DIRECTION_LABEL2ID.items()}


# ============================ CUSTOM MODELS ============================
class ContextPooler(nn.Module):
    """ContextPooler for DeBERTa-v2/v3, which takes the first token's hidden state."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_dim = config.hidden_size

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        pooled_output = self.dense(context_token)
        pooled_output = nn.GELU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return pooled_output


class SignalDirectionDeberta(DebertaV2PreTrainedModel):
    """DeBERTa-v3 model with two classification heads for signal and direction detection."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels_signal = 2
        self.num_labels_direction = len(DIRECTION_LABEL2ID)

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.signal_head = nn.Linear(self.pooler.output_dim, self.num_labels_signal)
        self.direction_head = nn.Linear(
            self.pooler.output_dim, self.num_labels_direction
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_signal=None,
        labels_direction=None,
        **kwargs,
    ):
        encoder_out = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.pooler(encoder_out.last_hidden_state)
        pooled_output = self.dropout(pooled_output)

        logits_signal = self.signal_head(pooled_output)
        logits_direction = self.direction_head(pooled_output)

        loss = None
        if labels_signal is not None and labels_direction is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_signal = loss_fct(
                logits_signal.view(-1, self.num_labels_signal), labels_signal.view(-1)
            )
            loss_direction = loss_fct(
                logits_direction.view(-1, self.num_labels_direction),
                labels_direction.view(-1),
            )
            loss = loss_signal + loss_direction

        return {
            "loss": loss,
            "logits_signal": logits_signal,
            "logits_direction": logits_direction,
        }


# ============================ CUSTOM TRAINERS ============================
class ClassifierTrainer(Trainer):
    """Custom Trainer for the dual-head SignalDirectionDeberta model."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels_signal = inputs.pop("labels_signal", None)
        labels_direction = inputs.pop("labels_direction", None)
        outputs = model(
            **inputs,
            labels_signal=labels_signal,
            labels_direction=labels_direction,
        )
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


# ============================ MAIN AI CLASS ============================
class AIClassifier:
    """AI system for trading signal extraction using DeBERTa and XLM-RoBERTa."""

    def __init__(
        self,
        classifier_model_name: str = "microsoft/deberta-v3-base",
        ner_model_name: str = "xlm-roberta-base",
        db_path: str = "messages.db",
        db_manager: Optional[DatabaseManager] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.classifier_model_name = classifier_model_name
        self.ner_model_name = ner_model_name
        self.preprocessor = TextPreprocessor()
        self.db_manager = (
            db_manager if db_manager is not None else DatabaseManager(db_path)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AIClassifier using device: {self.device}")

        self.classifier_tokenizer = DebertaV2Tokenizer.from_pretrained(
            classifier_model_name
        )

        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

        self.classifier_model: Optional[SignalDirectionDeberta] = None
        self.ner_model: Optional[AutoModelForTokenClassification] = None

        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = float(
                os.getenv("AI_CONFIDENCE_THRESHOLD", "0.7")
            )

    def _iter_training_rows(self) -> Iterable[Dict[str, Any]]:
        """Yields all labeled rows from the database for training."""
        try:
            raw_rows = self.db_manager.get_extended_labeled_data()
            for rr in raw_rows:
                r = dict(rr)
                for k in (
                    "direction",
                    "pair",
                    "stop_loss",
                    "leverage",
                    "targets",
                    "entry",
                ):
                    r.setdefault(k, None)
                yield r
        except Exception:
            logger.exception("Error fetching rows for training")
            return

    def _direction_to_id(self, val: Any) -> int:
        """Maps various direction representations to a consistent ID."""
        if val is None:
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
        """Creates weakly-supervised character-level spans for NER entities."""
        spans: List[Tuple[int, int, str]] = []
        lower_text = text.lower()

        def find_and_add_spans(value: Any, label: str):
            if not value:
                return
            value_str = str(value).strip().lower()
            if not value_str:
                return
            start_index = 0
            while (start_index := lower_text.find(value_str, start_index)) != -1:
                end_index = start_index + len(value_str)
                spans.append((start_index, end_index, label))
                start_index = end_index

        # Find spans for each entity type
        if row.get("pair"):
            find_and_add_spans(row["pair"], "PAIR")
        if row.get("stop_loss"):
            find_and_add_spans(row["stop_loss"], "STOP_LOSS")
        if row.get("entry"):
            find_and_add_spans(row["entry"], "ENTRY")
        if row.get("leverage"):
            find_and_add_spans(str(row["leverage"]).replace("x", ""), "LEVERAGE")

        targets = row.get("targets")
        if targets:
            try:
                target_list = (
                    json.loads(targets) if isinstance(targets, str) else targets
                )
                for target in target_list:
                    find_and_add_spans(target, "TAKE_PROFIT")
            except (json.JSONDecodeError, TypeError):
                pass

        # Sort and merge overlapping spans, preferring the longest match
        spans.sort(key=lambda s: (s[0], s[0] - s[1]))
        merged: List[Tuple[int, int, str]] = []
        for s in spans:
            if not merged or s[0] >= merged[-1][1]:
                merged.append(s)
        return merged

    def _align_ner_labels_to_tokens(
        self, encodings, spans: List[Tuple[int, int, str]]
    ) -> List[int]:
        """Aligns character-level spans to token-level labels for NER."""
        labels = [NER_LABEL2ID["O"]] * len(encodings["input_ids"])
        for start_char, end_char, label in spans:
            token_start_index = encodings.char_to_token(start_char)
            token_end_index = encodings.char_to_token(end_char - 1)

            if token_start_index is None or token_end_index is None:
                continue

            labels[token_start_index] = NER_LABEL2ID[f"B-{label}"]
            for i in range(token_start_index + 1, token_end_index + 1):
                labels[i] = NER_LABEL2ID[f"I-{label}"]

        # Use -100 for special tokens so they are ignored in the loss function
        for i, word_id in enumerate(encodings.word_ids()):
            if word_id is None:
                labels[i] = -100
        return labels

    def _build_classifier_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        """Builds a dataset for the SignalDirectionDeberta model."""
        data = {
            "input_ids": [],
            "attention_mask": [],
            "labels_signal": [],
            "labels_direction": [],
        }
        for r in rows:
            text = self.preprocessor.clean_text(r.get("message", ""))
            if not text:
                continue

            enc = self.classifier_tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )
            data["input_ids"].append(enc["input_ids"].squeeze().tolist())
            data["attention_mask"].append(enc["attention_mask"].squeeze().tolist())
            data["labels_signal"].append(int(r.get("is_signal") or 0))
            data["labels_direction"].append(self._direction_to_id(r.get("direction")))
        return Dataset.from_dict(data)

    def _build_ner_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        """Builds a dataset for the XLM-RoBERTa NER model."""
        data = {"input_ids": [], "attention_mask": [], "labels": []}
        for r in rows:
            text = self.preprocessor.clean_text(r.get("message", ""))
            if not text or not int(r.get("is_signal") or 0):
                continue

            # Get encoding for alignment without tensors
            enc_no_tensors = self.ner_tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_offsets_mapping=True,
            )
            spans = self._build_entity_spans(text, r)
            labels = self._align_ner_labels_to_tokens(enc_no_tensors, spans)

            # Get tensors for dataset
            enc = self.ner_tokenizer(
                text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )
            data["input_ids"].append(enc["input_ids"].squeeze().tolist())
            data["attention_mask"].append(enc["attention_mask"].squeeze().tolist())
            data["labels"].append(labels)
        return Dataset.from_dict(data)

    def _create_training_args(
        self, output_dir: str, epochs: int, lr: float
    ) -> TrainingArguments:
        """Creates a common set of TrainingArguments."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=2,  # Reduced to 2 for memory
            per_device_eval_batch_size=2,  # Reduced to 2 for memory
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,  # Disabled to avoid backward errors
            dataloader_num_workers=0,  # Disable multiprocessing to avoid memory issues
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_total_limit=1,
            report_to=[],
        )

    def _train_classifier(self, output_dir: str, rows: List[Dict[str, Any]]) -> bool:
        """Trains the signal and direction classification model."""
        logger.info("Starting classifier training...")
        classifier_output_dir = str(Path(output_dir) / "classifier")

        y = [int(r.get("is_signal") or 0) for r in rows]
        stratify = y if len(set(y)) > 1 else None
        train_rows, val_rows = train_test_split(
            rows, test_size=0.2, random_state=42, stratify=stratify
        )

        train_ds = self._build_classifier_dataset(train_rows)
        val_ds = self._build_classifier_dataset(val_rows)

        model = SignalDirectionDeberta.from_pretrained(self.classifier_model_name)
        training_args = self._create_training_args(
            classifier_output_dir, epochs=2, lr=2e-5
        )

        trainer = ClassifierTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model()
        self.classifier_tokenizer.save_pretrained(classifier_output_dir)
        logger.info(
            f"Classifier training completed and saved to {classifier_output_dir}"
        )
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _train_ner(self, output_dir: str, rows: List[Dict[str, Any]]) -> bool:
        """Trains the NER model on signal messages only."""
        logger.info("Starting NER model training...")
        ner_output_dir = str(Path(output_dir) / "ner")

        train_rows, val_rows = train_test_split(rows, test_size=0.2, random_state=42)
        train_ds = self._build_ner_dataset(train_rows)
        val_ds = self._build_ner_dataset(val_rows)

        model = AutoModelForTokenClassification.from_pretrained(
            self.ner_model_name,
            num_labels=len(NER_LABELS),
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID,
        )
        training_args = self._create_training_args(ner_output_dir, epochs=3, lr=3e-5)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        trainer.train()
        trainer.save_model()
        self.ner_tokenizer.save_pretrained(ner_output_dir)
        logger.info(f"NER training completed and saved to {ner_output_dir}")
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def train_model(self, output_dir: str = "./ai_model") -> bool:
        """Orchestrates the training of both classifier and NER models."""
        try:
            rows = list(self._iter_training_rows())
            if len(rows) < 20:
                logger.warning(
                    f"Training skipped: need at least 20 samples, got {len(rows)}"
                )
                return False

            classifier_success = self._train_classifier(output_dir, rows)

            signal_rows = [r for r in rows if int(r.get("is_signal") or 0) == 1]
            if len(signal_rows) < 10:
                logger.warning(
                    f"NER training skipped: need at least 10 signal samples, got {len(signal_rows)}"
                )
                return classifier_success

            ner_success = self._train_ner(output_dir, signal_rows)

            if classifier_success and ner_success:
                self.load_model(output_dir)
            return classifier_success and ner_success
        except Exception:
            logger.exception("Model training failed")
            return False

    def load_model(self, model_path: str = "./ai_model") -> bool:
        """Loads both the classifier and NER models from specified paths."""
        base_path = Path(model_path)
        classifier_path = base_path / "classifier"
        ner_path = base_path / "ner"

        classifier_loaded, ner_loaded = False, False

        try:
            if classifier_path.exists() and (classifier_path / "config.json").exists():
                self.classifier_model = SignalDirectionDeberta.from_pretrained(
                    str(classifier_path), local_files_only=True
                )
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                    str(classifier_path), local_files_only=True
                )
                self.classifier_model.to(self.device)
                logger.info(f"Classifier loaded from {classifier_path}")
                classifier_loaded = True
            else:
                logger.warning(f"Classifier model not found at {classifier_path}")
        except Exception:
            logger.exception(f"Error loading classifier model from {classifier_path}")

        try:
            if ner_path.exists() and (ner_path / "config.json").exists():
                # Free memory before loading NER model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.ner_model = AutoModelForTokenClassification.from_pretrained(
                    str(ner_path), local_files_only=True
                )
                self.ner_tokenizer = AutoTokenizer.from_pretrained(
                    str(ner_path), local_files_only=True
                )
                # Keep NER on CPU to avoid GPU OOM
                self.ner_model.to(torch.device("cpu"))
                logger.info(f"NER model loaded from {ner_path} (on CPU)")
                ner_loaded = True
            else:
                logger.warning(f"NER model not found at {ner_path}")
        except Exception:
            logger.exception(f"Error loading NER model from {ner_path}")

        return classifier_loaded and ner_loaded

    def predict(self, text: str) -> Tuple[int, float]:
        """Predicts if a text is a signal and returns the label and confidence."""
        if not self.classifier_model:
            logger.warning("Predict called before classifier model is loaded.")
            return 0, 0.0

        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return 0, 0.0

        enc = self.classifier_tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self.classifier_model.eval()
        with torch.no_grad():
            out = self.classifier_model(**enc)
            probs = torch.softmax(out["logits_signal"], dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            return int(pred.item()), float(conf.item())

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float]]:
        """Performs batch prediction for signal detection."""
        if not self.classifier_model or not texts:
            return [(0, 0.0)] * len(texts)

        cleaned_texts = [self.preprocessor.clean_text(t or "") for t in texts]
        enc = self.classifier_tokenizer(
            cleaned_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        self.classifier_model.eval()
        with torch.no_grad():
            out = self.classifier_model(**enc)
            probs = torch.softmax(out["logits_signal"], dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            return list(zip(pred.cpu().tolist(), conf.cpu().tolist()))

    def _decode_ner_predictions(self, encodings, logits) -> Dict[str, List[Any]]:
        """Decodes NER logits into a dictionary of entities."""
        predictions = torch.argmax(logits, dim=2)[0].cpu().tolist()
        tokens = self.ner_tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])

        entities, current_entity = {}, []
        for i, pred_id in enumerate(predictions):
            word_id = encodings.word_ids(batch_index=0)[i]
            if word_id is None:
                continue  # Skip special tokens

            label = self.ner_model.config.id2label[pred_id]
            if label.startswith("B-"):
                if current_entity:
                    entities.setdefault(current_entity[0], []).append(
                        "".join(t for _, t in current_entity[1:]).replace(" ", "")
                    )
                current_entity = [label[2:], (i, tokens[i])]
            elif (
                label.startswith("I-")
                and current_entity
                and label[2:] == current_entity[0]
            ):
                current_entity.append((i, tokens[i]))
            else:
                if current_entity:
                    entities.setdefault(current_entity[0], []).append(
                        "".join(t for _, t in current_entity[1:]).replace(" ", "")
                    )
                current_entity = []
        if current_entity:
            entities.setdefault(current_entity[0], []).append(
                "".join(t for _, t in current_entity[1:]).replace(" ", "")
            )
        return entities

    def extract_signal_fields(self, text: str) -> Dict[str, Any]:
        """Full pipeline: classify, then extract NER if it's a signal."""
        if not self.classifier_model or not self.ner_model:
            logger.warning("Models not loaded for extraction.")
            return {}

        clean_text = self.preprocessor.clean_text(text)
        if not clean_text:
            return {}

        # Step 1: Classification
        classifier_enc = self.classifier_tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        classifier_enc = {k: v.to(self.device) for k, v in classifier_enc.items()}

        self.classifier_model.eval()
        with torch.no_grad():
            classifier_out = self.classifier_model(**classifier_enc)

        signal_probs = torch.softmax(classifier_out["logits_signal"], dim=-1)
        signal_conf, signal_pred = torch.max(signal_probs, dim=-1)

        result = {
            "is_signal": bool(signal_pred.item()),
            "confidence": round(signal_conf.item(), 4),
            "direction": None,
            "pair": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": [],
            "leverage": None,
        }

        if not result["is_signal"]:
            return result

        # Step 2: Extract Direction
        dir_pred = torch.argmax(classifier_out["logits_direction"], dim=-1).item()
        result["direction"] = DIRECTION_ID2LABEL.get(dir_pred, "none")

        # Step 3: Extract NER entities
        ner_enc = self.ner_tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        # Send NER tensors to the same device as the NER model (kept on CPU to avoid OOM)
        ner_device = next(self.ner_model.parameters()).device
        ner_enc = {k: v.to(ner_device) for k, v in ner_enc.items()}

        self.ner_model.eval()
        with torch.no_grad():
            ner_out = self.ner_model(**ner_enc)

        entities = self._decode_ner_predictions(
            {k: v.to("cpu") for k, v in ner_enc.items()}, ner_out.logits.to("cpu")
        )

        # Step 4: Populate result with cleaned entities
        result["pair"] = entities.get("PAIR", [None])[0]
        try:
            result["entry"] = float(entities.get("ENTRY", [None])[0])
        except (TypeError, ValueError):
            pass
        try:
            result["stop_loss"] = float(entities.get("STOP_LOSS", [None])[0])
        except (TypeError, ValueError):
            pass
        try:
            result["leverage"] = float(entities.get("LEVERAGE", [None])[0])
        except (TypeError, ValueError):
            pass
        try:
            result["take_profit"] = [
                float(tp) for tp in entities.get("TAKE_PROFIT", [])
            ]
        except (TypeError, ValueError):
            pass

        return result

    def get_uncertain_samples(
        self, texts: List[str], n_samples: int
    ) -> List[Tuple[int, str, float]]:
        """Identifies samples with low prediction confidence for active learning."""
        if not self.classifier_model or not texts:
            return []

        predictions = self.predict_batch(texts)
        uncertain_samples = [
            (i, texts[i], conf)
            for i, (_, conf) in enumerate(predictions)
            if conf < self.confidence_threshold
        ]
        uncertain_samples.sort(key=lambda x: x[2])
        return uncertain_samples[:n_samples]


class ActiveLearningManager:
    """Manages the active learning workflow by suggesting messages for labeling."""

    def __init__(self, classifier: AIClassifier, db_manager: DatabaseManager):
        self.classifier = classifier
        self.db_manager = db_manager

    def suggest_next_messages(self, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Requesting {n_suggestions} suggestions for active learning.")
        try:
            unlabeled_messages = self.db_manager.get_unlabeled_messages(limit=500)
            if not unlabeled_messages:
                return []

            message_texts = [msg["message"] for msg in unlabeled_messages]
            uncertain_indices = {
                idx
                for idx, _, _ in self.classifier.get_uncertain_samples(
                    message_texts, n_suggestions
                )
            }

            return [
                dict(unlabeled_messages[i]) for i in sorted(list(uncertain_indices))
            ]
        except Exception:
            logger.exception("Failed to suggest next messages for labeling.")
            return []

    def get_training_recommendations(self) -> Dict[str, Any]:
        try:
            return self.db_manager.get_training_stats()
        except Exception:
            logger.exception("Failed to get training recommendations.")
            return {}
