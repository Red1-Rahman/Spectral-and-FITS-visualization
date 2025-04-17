from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analyzer/', include('analyzer.urls')),
    path('', include('analyzer.urls')), # Make the analyzer the default page
]