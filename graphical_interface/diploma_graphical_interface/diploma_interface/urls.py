from django.urls import path
from diploma_interface import views
 
urlpatterns = [
    path('', views.index, name='index'),
    path('import-csv/', views.get_info_from_csv, name='import-csv'),
    path('analysis/', views.get_data_analysis, name='analysis'),
    path('get_subjects_intersection/', views.get_subjects_intersection, name='get_subjects_intersection'),
    path('download/<path:filename>/', views.download_analysis_model, name='download_analysis_model'),
    path('prediction/', views.predition_page, name='predition_page'),
    path('info/', views.get_info, name='info'),
]