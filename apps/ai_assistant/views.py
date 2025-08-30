import os
import json
import logging
import asyncio
from typing import Optional, Tuple
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from dotenv import load_dotenv

from apps.telegram_client.message_extractor import MessageExtractor
from utils.config import DB_PATH as CONFIG_DB_PATH, MODEL_DIR
from . import services

# Load environment variables
load_dotenv()

# Import credentials from environment variables
try:
    api_id_str = os.getenv("API_ID")
    API_ID = int(api_id_str) if api_id_str else None
    API_HASH = os.getenv("API_HASH")
except (ValueError, TypeError):
    API_ID, API_HASH = None, None


logger = logging.getLogger(__name__)

DB_PATH: str = str(CONFIG_DB_PATH)


# !CHECK AI GENERATED BULLSHIT
def _ensure_services():
    if (
        services.db_manager is None
        or services.ai_classifier is None
        or services.al_manager is None
    ):
        services.init_services(DB_PATH)


# !CHECK AI GENERATED BULLSHIT
def _check_creds() -> Tuple[HttpResponse | int, str | None]:
    """Validate and return Telegram API credentials from environment variables."""
    if not API_ID or not API_HASH:
        return redirect(
            "/ai/?msg="
            + "Missing API credentials in environment variables (API_ID/API_HASH)"
        ), None

    return API_ID, API_HASH


# !CHECK AI GENERATED BULLSHIT
def dashboard(request: HttpRequest) -> HttpResponse:
    _ensure_services()
    db = services.db_manager
    alm = services.al_manager
    stats = db.get_labeling_stats()
    channels = db.get_channel_stats()
    training = alm.get_training_recommendations()
    message = request.GET.get("msg")
    return render(
        request,
        "ai_dashboard.html",
        {
            "stats": stats,
            "channels": channels,
            "training": training,
            "message": message,
        },
    )


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["POST"])
def refresh_channel_names(request: HttpRequest) -> HttpResponse:
    api_id, api_hash = _check_creds()
    if isinstance(api_id, HttpResponse):
        return api_id
    assert api_hash is not None
    extractor = MessageExtractor(db_path=DB_PATH)
    try:
        updated = asyncio.run(extractor.backfill_channel_metadata(api_id, api_hash))
        return redirect("/ai/?msg=" + f"Updated {updated} channels")
    except Exception as e:
        return redirect("/ai/?msg=" + f"Error: {e}")


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["GET", "POST"])
def extract_messages(request: HttpRequest) -> HttpResponse:
    message: Optional[str] = None
    if request.method == "POST":
        api_id, api_hash = _check_creds()
        if isinstance(api_id, HttpResponse):
            return api_id
        assert api_hash is not None
        limit_str = request.POST.get("limit")
        limit = int(limit_str) if limit_str else None
        extractor = MessageExtractor(db_path=DB_PATH)
        try:
            import asyncio

            asyncio.run(
                extractor.extract_messages_from_folder(int(api_id), api_hash, limit)
            )
            message = "Extraction completed"
        except Exception as e:
            message = f"Error: {e}"
    return render(request, "extract.html", {"message": message})


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["POST"])
def extract_single(request: HttpRequest) -> HttpResponse:
    api_id, api_hash = _check_creds()
    if isinstance(api_id, HttpResponse):
        return api_id
    assert api_hash is not None
    entity = request.POST.get("entity")
    limit_str = request.POST.get("limit")
    limit = int(limit_str) if limit_str else None
    message: Optional[str] = None
    extractor = MessageExtractor(db_path=DB_PATH)
    if not entity:
        return redirect("/ai/?msg=" + "Entity is required for single extraction")
    try:
        import asyncio

        asyncio.run(
            extractor.extract_messages_from_channel(
                int(api_id), api_hash, entity, limit
            )
        )
        message = f"Extraction completed for {entity}"
    except Exception as e:
        message = f"Error: {e}"
    return render(request, "extract.html", {"message": message})


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["GET", "POST"])
def train_model(request: HttpRequest) -> HttpResponse:
    status: Optional[str] = None
    if request.method == "POST":
        _ensure_services()
        clf = services.ai_classifier
        ok = clf.train_model(output_dir=str(MODEL_DIR))
        status = (
            "Training started and finished successfully"
            if ok
            else "Training failed or not enough data"
        )
    return render(request, "train.html", {"status": status})


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["GET"])
def label(request: HttpRequest) -> HttpResponse:
    """Labeling over unlabeled messages (messages table) with round-robin across channels.
    On save, entries are written to the labeled table.
    """
    _ensure_services()
    db = services.db_manager
    clf = services.ai_classifier

    # Try to load model once if not already loaded
    try:
        if clf.classifier_model is None or clf.ner_model is None:
            model_loaded = clf.load_model(str(MODEL_DIR))
            if not model_loaded:
                logger.warning("Failed to load AI models for extraction")
    except Exception as e:
        logger.exception(f"Error loading AI models: {e}")

    # Round-robin selection across available channels
    channels = db.get_available_channels()

    item = None
    existing_label = None
    is_relabeling = False

    # For template compatibility; sequential mode not used in this flow
    sequential = False
    seq_total = None
    seq_done = None
    seq_current_index = None

    if channels:
        session_key = "rr_last_channel_id_extended"
        last_id = request.session.get(session_key)
        start_idx = 0
        if last_id in channels:
            try:
                start_idx = (channels.index(last_id) + 1) % len(channels)
            except ValueError:
                start_idx = 0

        # Try each channel once to find an unlabeled message
        for offset in range(len(channels)):
            channel_id = channels[(start_idx + offset) % len(channels)]
            candidate = db.get_random_unlabeled_message_from_channel(
                channel_id=channel_id
            )
            if candidate is not None:
                item = candidate  # row from messages table
                existing_label = None
                is_relabeling = False
                request.session[session_key] = channel_id
                break

        # If none found, clear pointer
        if item is None:
            request.session.pop(session_key, None)

    ai = None
    ai_extracted = None
    if item is not None:
        try:
            # sqlite3.Row requires dict-style access
            text = item["message"]
            pred, conf = clf.predict(text)
            if conf and conf > 0:
                ai = {
                    "label": "Signal" if int(pred) == 1 else "Not signal",
                    "confidence": int(round(float(conf) * 100)),
                    "pred_value": int(pred),
                }
            # Log prediction to console (even if low confidence)
            preview = (text or "")[:200].replace("\n", " ")
            try:
                logger.info(
                    f"UI Predict (extended): channel={item['channel_id']} id={item['id']} pred={int(pred)} conf={float(conf):.3f} text='{preview}'"
                )
            except Exception:
                logger.info(
                    f"UI Predict (extended): channel={item['channel_id']} id={item['id']} text='{preview}'"
                )

            # Always attempt to extract fields so Signal click can autocomplete
            try:
                ai_extracted = clf.extract_signal_fields(text)
                logger.info(
                    f"UI Extract (extended): channel={item['channel_id']} id={item['id']} extracted={ai_extracted}"
                )
                # Add debug logging for template data
                if ai_extracted:
                    logger.info(f"AI extracted data for template: {ai_extracted}")
                    # Convert targets list to JSON string for JavaScript consumption
                    if ai_extracted.get("targets") and isinstance(
                        ai_extracted["targets"], list
                    ):
                        ai_extracted["targets_json"] = json.dumps(
                            ai_extracted["targets"]
                        )
                else:
                    logger.warning("AI extraction returned empty result")
            except Exception as e:
                logger.exception(f"Error during AI extraction: {e}")
                ai_extracted = None
        except Exception:
            ai = None
            ai_extracted = None

    return render(
        request,
        "extended_label.html",
        {
            "item": item,
            "ai": ai,
            "ai_extracted": ai_extracted,
            "existing_label": existing_label,
            "is_relabeling": is_relabeling,
            "sequential": sequential,
            "seq_total": seq_total,
            "seq_done": seq_done,
            "seq_current_index": seq_current_index,
        },
    )


# !CHECK AI GENERATED BULLSHIT
@require_http_methods(["POST"])
def save_label(request: HttpRequest) -> HttpResponse:
    """Save label with all signal fields."""
    _ensure_services()
    db = services.db_manager

    message_id = int(request.POST["message_id"])
    channel_id = int(request.POST["channel_id"])
    message = request.POST["message"]
    is_signal = request.POST["is_signal"] == "1"
    sequential = request.POST.get("sequential") in ("1", "true", "yes", "on")
    labeled_row_id = request.POST.get("labeled_id")
    labeled_row_id_int = (
        int(labeled_row_id) if labeled_row_id and labeled_row_id.isdigit() else None
    )

    # Extended fields - only process if is_signal is True
    direction = None
    pair = None
    stop_loss = None
    leverage = None
    targets = None
    entry = None

    if is_signal:
        # Parse direction
        direction_str = request.POST.get("direction", "").strip()
        if direction_str:
            direction = int(direction_str)

        # Parse pair
        pair = request.POST.get("pair", "").strip() or None

        # Parse numeric fields
        try:
            stop_loss_str = request.POST.get("stop_loss", "").strip()
            if stop_loss_str:
                stop_loss = float(stop_loss_str)
        except (ValueError, TypeError):
            stop_loss = None

        try:
            leverage_str = request.POST.get("leverage", "").strip()
            if leverage_str:
                leverage = float(leverage_str)
        except (ValueError, TypeError):
            leverage = None

        try:
            entry_str = request.POST.get("entry", "").strip()
            if entry_str:
                entry = float(entry_str)
        except (ValueError, TypeError):
            entry = None

        # Parse targets array
        targets_str = request.POST.get("targets", "").strip()
        if targets_str:
            try:
                # Parse comma-separated values
                targets_list = [
                    float(x.strip()) for x in targets_str.split(",") if x.strip()
                ]
                if targets_list:
                    targets = json.dumps(targets_list)
            except (ValueError, TypeError):
                targets = None

    # If we are relabeling an existing labeled row (sequential or existing_label present), update by id
    if labeled_row_id_int is not None:
        try:
            db.update_label_by_id(
                labeled_id=labeled_row_id_int,
                is_signal=is_signal,
                direction=direction,
                pair=pair,
                stop_loss=stop_loss,
                leverage=leverage,
                targets=targets,
                entry=entry,
            )
        except Exception:
            logger.exception(
                "Failed to update existing labeled row; falling back to upsert by message_id"
            )
            db.save_label(
                message_id=message_id,
                channel_id=channel_id,
                message=message,
                is_signal=is_signal,
                direction=direction,
                pair=pair,
                stop_loss=stop_loss,
                leverage=leverage,
                targets=targets,
                entry=entry,
            )
    else:
        # No existing labeled row; create or replace by message_id
        db.save_label(
            message_id=message_id,
            channel_id=channel_id,
            message=message,
            is_signal=is_signal,
            direction=direction,
            pair=pair,
            stop_loss=stop_loss,
            leverage=leverage,
            targets=targets,
            entry=entry,
        )

    # Advance sequential pointer if in sequential mode and we were relabeling an existing row
    if sequential and labeled_row_id_int is not None:
        try:
            db.set_app_state("extended_seq_last_id", str(labeled_row_id_int))
        except Exception:
            logger.exception("Failed to update sequential pointer for labeling")

    # Preserve mode on redirect
    if sequential:
        return redirect("/ai/label?sequential=1")
    else:
        return redirect("/ai/label/")
