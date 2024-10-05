from django.db import models

class Input(models.Model):
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name
    
class Output(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE)
    value = models.CharField(max_length=2000)

    def __str__(self):
        return self.name