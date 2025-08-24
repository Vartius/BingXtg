from django.apps import apps as django_apps

# Ensure Django uses our AppConfig with startup initialization (harmless on Django >=3.2)
default_app_config = "webapp.apps.WebappConfig"


def get_services():
    """Return (db_manager, ai_classifier, active_learning_manager) singletons."""
    app_config = django_apps.get_app_config("webapp")
    # Attributes are set in WebappConfig.ready()
    return (
        getattr(app_config, "db_manager", None),
        getattr(app_config, "ai_classifier", None),
        getattr(app_config, "al_manager", None),
    )
