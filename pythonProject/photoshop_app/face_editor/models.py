from django.contrib.auth.models import User
from django.db import models


class UserPhoto(models.Model):
    photo = models.ImageField(upload_to='photos/')


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    face_image = models.ImageField(upload_to='user_faces/')
