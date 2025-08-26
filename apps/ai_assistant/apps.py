from django.apps import AppConfig


class AiAssistantConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.ai_assistant"
    verbose_name = "AI Assistant"

    def ready(self):
        # Avoid double initialization when using Django's autoreloader in debug
        import os
        from django.conf import settings

        if getattr(settings, "DEBUG", False) and os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from utils.config import DB_PATH
            from . import services

            services.init_services(str(DB_PATH))
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Failed to initialize services at startup"
            )
