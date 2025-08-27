from django.urls import path
from django.http import HttpResponse
from . import views

# !CHECK AI GENERATED BULLSHIT


def favicon_view(request):
    return HttpResponse(status=204)  # No Content


urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("data/", views.dashboard_data, name="dashboard_data"),
    path("api/dashboard-data/", views.dashboard_data, name="api_dashboard_data"),
    path("favicon.ico", favicon_view, name="favicon"),
]
