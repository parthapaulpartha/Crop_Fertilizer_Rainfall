from django.db import models

# Create your models here.
#------- Contact us -------

class contact(models.Model):
    name = models.CharField(max_length=50, default='')
    email = models.EmailField(max_length=50, default='')
    message = models.TextField(max_length=400, default='')
