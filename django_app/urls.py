from django.urls import path

from . import views


urlpatterns = [
    path('index/', views.index, name='index'),
    path('datamovie/', views.showDataMovie, name='datamovie'),
    path('datarating/', views.showDataRating, name='datarating'),
    path('jsonrating/', views.showRatingJson, name='jsonrating'),
    path('jsonmovie/', views.showMovieJson, name='jsonmovie'),
    path('insert/', views.insertRating, name='insert'),
    path('delete/', views.deleteRating, name='delete'),
    path('update/', views.updateRating, name='update'),
    path('predict/', views.predictImage, name='predict'),

]