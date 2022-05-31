from django.urls import path
from app.consumer import *

ws_urlpatterns = [
    path('train/<str:model_name>/output/', training_consumer.as_asgi()),
    path('predict/<str:model_name>/output/', predict_consumer.as_asgi()),
    path('backtest/output/', backtest_consumer.as_asgi())
]