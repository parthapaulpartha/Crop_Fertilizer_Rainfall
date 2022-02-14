from django.contrib import admin
from .models import contact

# Register your models here.

#---- admin header --------
admin.site.site_header = 'Crop Fertilizer Rainfall'

#------ Contact -------
class ContactAdmin(admin.ModelAdmin):
    list_display = 'name', 'email', 'message'
    list_filter = 'name', 'email', 'message'

admin.site.register(contact, ContactAdmin)
