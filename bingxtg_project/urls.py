from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.trading_bot.urls")),
    path("ai/", include("apps.ai_assistant.urls")),
    path("telegram/", include("apps.telegram_client.urls")),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
