from django.urls import path
from .views import main
from .views import ExtractImagesFromPdfView, ClassifyImagesView
urlpatterns = [
    path('home', main),
    path('extract-images/', ExtractImagesFromPdfView.as_view(), name='extract-images'),
    path('classify-images/', ClassifyImagesView.as_view(), name='classify-images'),

]