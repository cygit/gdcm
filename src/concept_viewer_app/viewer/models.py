from django.db import models

# Create your models here.


class Result(models.Model):
    key = models.CharField(max_length=512, primary_key=True)
    dataset = models.CharField(max_length=64)
    grid_path = models.CharField(max_length=256)
    run_id = models.CharField(max_length=64)
    epoch = models.IntegerField()
    window_size = models.IntegerField()
    embed_dim = models.IntegerField()
    nnegs = models.IntegerField()
    nconcepts = models.IntegerField()
    lam = models.FloatField()
    rho = models.FloatField()
    eta = models.FloatField()
    topics = models.TextField(default='')
    coherence_per_topic = models.TextField(default='')
    coherence = models.FloatField()
    total_loss = models.FloatField()
    avg_sgns_loss = models.FloatField()
    avg_dirichlet_loss = models.FloatField()
    avg_pred_loss = models.FloatField()
    avg_div_loss = models.FloatField()
    train_auc = models.FloatField()
    test_auc = models.FloatField()

    def __str__(self):
        return '{}:rho{}'.format(self.key, self.rho)
