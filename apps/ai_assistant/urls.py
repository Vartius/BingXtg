from django.urls import path
from . import views
# !CHECK AI GENERATED BULLSHIT

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("extract/", views.extract_messages, name="extract_messages"),
    path("extract/single/", views.extract_single, name="extract_single"),
    path("label/", views.label, name="label"),
    path("label/save/", views.save_label, name="save_label"),
    path("train/", views.train_model, name="train_model"),
    path(
        "channels/refresh/", views.refresh_channel_names, name="refresh_channel_names"
    ),
]
