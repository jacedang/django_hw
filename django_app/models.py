from django.db import models

# Create your models here.
class Movies(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=200)
    genres = models.CharField(max_length=100)
    class Meta:
        db_table="movies"

class Ratings(models.Model):
    user_id = models.IntegerField()
    # movie_id = models.ForeignKey(
    #     'Movies',
    #     on_delete=models.CASCADE,
    # )
    movie_id = models.IntegerField()
    rating = models.FloatField()
    class Meta:
        db_table="ratings"