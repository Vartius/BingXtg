from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("bot.urls")),
    path("ai/", include("webapp.urls")),  # Added routes for the AI webapp
]
