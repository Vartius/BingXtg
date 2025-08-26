from __future__ import annotations
from typing import Any

db_manager: Any = None
ai_classifier: Any = None
al_manager: Any = None


def init_services(db_path: str) -> None:
    """Initialize singletons once. Safe to call multiple times."""
    global db_manager, ai_classifier, al_manager
    if db_manager is not None and ai_classifier is not None and al_manager is not None:
        return

    from utils.database_manager import DatabaseManager
    from utils.ai_assistant import AIClassifier, ActiveLearningManager

    db_manager = DatabaseManager(db_path)
    # Ensure tables exist at startup
    db_manager.init_database()

    ai_classifier = AIClassifier(db_path=db_path)
    al_manager = ActiveLearningManager(ai_classifier, db_manager)
