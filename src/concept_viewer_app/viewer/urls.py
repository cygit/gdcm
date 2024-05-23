from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('update', views.update_table),
    path('all', views.get_result),
]
