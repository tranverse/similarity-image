from django.db import models

class Research(models.Model):
    image_id = models.AutoField(primary_key=True)
    image_field_name = models.CharField(max_length=500, null=True, blank=True)
    doi = models.CharField(max_length=255, null=True, blank=True)
    title = models.CharField(max_length=500, null=True, blank=True)
    caption = models.CharField(max_length=1000, null=True, blank=True)
    page_number = models.IntegerField(null=True, blank=True)
    extraction_date = models.DateTimeField(null=True, blank=True)
    authors = models.CharField(max_length=500, null=True, blank=True)
    approved_date = models.CharField(max_length=50, null=True, blank=True)
    language = models.CharField(max_length=2, choices=[('vi', 'vietnamese'), ('en', 'English')], default='vi')
    class_name = models.CharField(max_length=100, null=True, blank=True)
    class Meta:
        db_table = 'research'
        managed = False

    def __str__(self):
        return self.title if self.title else "No Title"

class Feature(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ForeignKey(Research, on_delete=models.CASCADE, related_name='features')
    model_name = models.CharField(max_length=50)
    feature_vector = models.BinaryField()

    class Meta:
        db_table = 'feature'