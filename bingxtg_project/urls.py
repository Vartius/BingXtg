from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
# !CHECK AI GENERATED BULLSHIT

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.trading_bot.urls")),
    path("ai/", include("apps.ai_assistant.urls")),
    path("telegram/", include("apps.telegram_client.urls")),
]

# Serve media and static files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(
        settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0]
    )
    # Also serve from static root if it exists
    import os

    if os.path.exists(settings.STATIC_ROOT):
        urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
