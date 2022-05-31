from django.urls import path
from app.views import *


urlpatterns = [
    path('', home),
    path('train/', train),
    path('train/<str:model_name>/', setting),
    path('train/<str:model_name>/output/', output),
    path('predict/', predict),
    path('predict/<str:model_name>/', predict_setting),
    path('predict/<str:model_name>/output/', predict_output),
    path('backtest/', backtest),
    path('backtest/output/', backtest_result),
    path('data/', data),
    path('data/<str:format_>/', return_data),
    path('resource/<str:filename>', resource)
]  