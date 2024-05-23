import json
import os
from collections import OrderedDict
import re

from concept_viewer_app.viewer.frex_coherence import get_coherence, load_dataset
import pandas as pd

ROOT = os.path.abspath(__file__ + '/..')
DATASET_KEYS = ['window_size']
GDCM_KEYS = ['embed_dim', 'nnegs', 'nconcepts', 'lam', 'rho', 'eta']
METRIC_KEYS = ['total_loss', 'avg_sgns_loss', 'avg_dirichlet_loss', 'avg_pred_loss', 'avg_div_loss', 'train_auc',
               'test_auc']


def get_fields(grid_path, metrics, params, dataset, run_id, epoch):
    fields = OrderedDict()
    run_id_path = os.path.join(grid_path, run_id)
    fields['key'] = '%s:%s:%s:%d' % (dataset, os.path.basename(grid_path), run_id, epoch)
    fields['dataset'] = dataset
    fields['grid_path'] = grid_path
    fields['run_id'] = run_id
    fields['epoch'] = epoch
    for key in DATASET_KEYS:
        fields[key] = params['dataset'][key]
    for key in GDCM_KEYS:
        fields[key] = params['gdcm'][key]
    concept_dir = os.path.join(run_id_path, "concept")
    topics_text = open(os.path.join(concept_dir, "epoch%d.txt" % epoch)).read()
    fields['topics'] = topics_text
    topics = [t.split(':')[1].strip().split(' ') for t in topics_text.split('\n') if t]
    _, corpus, dic = load_dataset(grid_path, dataset, run_id)
    fields['coherence_per_topic'], fields['coherence'] = get_coherence(topics, corpus, dic)
    fields.update(metrics.iloc[epoch][METRIC_KEYS].to_dict())
    return fields


def save_fixtures(grid_search_path):
    grid_search_path = os.path.abspath(grid_search_path)
    with open(os.path.join(grid_search_path, "config.json"), "r") as f:
        config = json.load(f)
    dataset = config["dataset"]
    data_list = []
    for run_id in os.listdir(grid_search_path):
        run_dir = os.path.join(grid_search_path, run_id)
        if not os.path.exists(os.path.join(run_dir, "result.json")):
            continue
        with open(os.path.join(run_dir, "params.json"), "r") as f:
            params = json.load(f)
            model_dir = os.path.join(run_dir, "model")
            metrics = pd.read_csv(os.path.join(run_dir, "train_metrics.txt"))
            for model_file in os.listdir(model_dir):
                epoch_match = re.search('epoch(\d+)\.pytorch', model_file)
                if epoch_match is None:
                    continue
                epoch = int(epoch_match.group(1))
                data_dict = OrderedDict()
                data_dict['model'] = 'viewer.result'
                data_dict['fields'] = get_fields(grid_search_path, metrics, params, dataset, run_id, epoch)
                data_list.append(data_dict)
    data_file = os.path.abspath(os.path.join(grid_search_path, 'concept_viewer_fixtures.json'))
    with open(data_file, 'w') as f:
        json.dump(data_list, f)
    return data_file
