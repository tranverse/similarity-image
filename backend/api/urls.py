from django.urls import path
from .views import main
from .views import ExtractImagesFromPdfView, ClassifyAndFindSimilarImagesView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Các URL khác
    path('home', main),
    path('extract-images/', ExtractImagesFromPdfView.as_view(), name='extract-images'),
    path('classify-images/', ClassifyAndFindSimilarImagesView.as_view(), name='classify-images'),
    # path('add-features/<str:image_file_name>/<str:model_name>/', AddFeatures.as_view(), name='add-feature'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
