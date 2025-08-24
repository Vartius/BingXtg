from django.apps import AppConfig


class WebappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "webapp"

    def ready(self):
        # Avoid double initialization when using Django's autoreloader in debug
        import os
        from django.conf import settings

        if getattr(settings, "DEBUG", False) and os.environ.get("RUN_MAIN") != "true":
            return

        try:
            from src.config import DB_PATH
            from . import services

            services.init_services(str(DB_PATH))
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Failed to initialize services at startup"
            )
