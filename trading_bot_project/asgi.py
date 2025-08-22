import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
import bot.routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trading_bot_project.settings")

# Get the default HTTP application
django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(
            URLRouter(bot.routing.websocket_urlpatterns)
        ),
    }
)
