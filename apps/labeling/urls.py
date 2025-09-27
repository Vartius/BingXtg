from django.urls import path

from .views import AutoLabelingTriggerView, LabelingDashboardView

app_name = "labeling"

urlpatterns = [
    path("", LabelingDashboardView.as_view(), name="index"),
    path("auto/", AutoLabelingTriggerView.as_view(), name="auto"),
]
