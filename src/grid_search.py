import hashlib
import itertools
import json
import os
import random
import sys
import threading
import time
import traceback
from queue import Queue

import gc
import pandas
import torch

from dataset.csv import CSVDataset
from gdcm import GuidedDiverseConceptMiner
from toolbox.helper_functions import get_dataset

GRID_ROOT = os.path.abspath(__file__ + '/../../grid_search')

random.seed(0)

torch.backends.cudnn.benchmark = True

done_ids = set()
queue = Queue()
lock = threading.Lock()


# TODO: multithread result file conflict
def training_thread(device_idx, ds, config):
    global results
    while True:
        device = devices[device_idx]
        dataset_params, gdcm_params, fit_params = queue.get()
        params = {"dataset": dataset_params, "gdcm": gdcm_params, "fit": fit_params}
        run_id = hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()
        try:
            with lock:
                run_dir = os.path.join(grid_dir, run_id)
                result_file = os.path.join(run_dir, "result.json")
                if os.path.exists(result_file):
                    print("Configuration {} has already been run, skip...".format(run_id))
                    queue.task_done()
                    continue
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, sort_keys=True)
                data_dict = ds.load_data(dataset_params)
                # remove gensim keys which are only used for visualization
                del data_dict["gensim_corpus"]
                del data_dict["gensim_dictionary"]
            print("Beginning training run on {}... with id {} dataset_params={}, gdcm_params={}, fit_params={}".format(
                device, run_id, dataset_params, gdcm_params, fit_params))
            print("Save grid search results to {}".format(os.path.abspath(run_dir)))
            start = time.perf_counter()
            if torch.cuda.is_available():
                fc_miner = GuidedDiverseConceptMiner(run_dir, gpu=gpus[device_idx], file_log=True,
                                               **gdcm_params, **data_dict)
            else:
                fc_miner = GuidedDiverseConceptMiner(run_dir, file_log=True, **gdcm_params, **data_dict)
            metrics = fc_miner.fit(**fit_params)
            # fc_miner.visualize()
            end = time.perf_counter()
            run_time = end - start
            del fc_miner
            del data_dict
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print("WARNING: exception raised while training on {} with dataset_params={}, "
                  "gdcm_params={}, fit_params={}".format(device, dataset_params, gdcm_params, fit_params))
        else:
            best_losses = metrics[:, :-2].min(axis=0)
            best_aucs = metrics[:, -2:].max(axis=0)
            best_metrics = {"id": run_id, "run_time": run_time,
                            "total_loss": best_losses[0], "sgns_loss": best_losses[1],
                            "dirichlet_loss": best_losses[2], "pred_loss": best_losses[3], "div_loss": best_losses[4],
                            "train_auc": best_aucs[0], "test_auc": best_aucs[1]}
            new_result = {**{"dataset." + k: v for k, v in dataset_params.items()},
                          **{"gdcm." + k: v for k, v in gdcm_params.items()},
                          **{"fit." + k: v for k, v in fit_params.items()},
                          **best_metrics}
            with open(result_file, 'w') as f:
                json.dump(new_result, f, sort_keys=True)
            with lock:
                results = results.append(new_result, ignore_index=True)
                results.to_csv(os.path.join(grid_dir, "results.csv"), index=False)
            print("Training run complete, results:", best_metrics)
        torch.cuda.empty_cache()
        gc.collect()
        queue.task_done()


def grid_search(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    global gpus, grid_dir, results, devices
    out_dir = config["out_dir"]
    max_threads = config["max_threads"]
    if torch.cuda.is_available():
        if "gpus" in config.keys():
            gpus = config["gpus"]
        else:
            gpus = list(range(torch.cuda.device_count()))
        devices = ["cuda:{}".format(i) for i in gpus]
    else:
        devices = ["cpu"]
    dataset_params = config["dataset_params"]
    gdcm_params = config["gdcm_params"]
    fit_params = config["fit_params"]

    results = pandas.DataFrame(columns=["id", "run_time", "total_loss", "sgns_loss", "dirichlet_loss",
                                        "pred_loss", "div_loss", "train_auc", "test_auc"]
                                       + ["dataset." + k for k in dataset_params.keys()]
                                       + ["gdcm." + k for k in gdcm_params.keys()]
                                       + ["fit." + k for k in fit_params.keys()])
    dataset_params_len = len(dataset_params.keys())
    gdcm_params_len = len(gdcm_params.keys())
    # generate every possible combinations of all possible dataset_params and gdcm_params
    combos = [(dict(zip(dataset_params.keys(), values[:dataset_params_len])),
               dict(zip(gdcm_params.keys(), values[dataset_params_len:dataset_params_len + gdcm_params_len])),
               dict(zip(fit_params.keys(), values[dataset_params_len + gdcm_params_len:])))
              for values in itertools.product(*dataset_params.values(), *gdcm_params.values(), *fit_params.values())]
    random.shuffle(combos)
    print("Start grid search with %d combos" % len(combos))
    if os.path.isfile(os.path.join(out_dir, "results.csv")):
        results = pandas.read_csv(os.path.join(out_dir, "results.csv"))
    for combo in combos:
        queue.put(combo)
    if config["dataset"] == 'csv':
        ds = CSVDataset(config["csv-path"], config["csv-text"], config["csv-label"])
        grid_dir = os.path.join(GRID_ROOT, "CSV_" + os.path.basename(config["csv-path"]), out_dir)
    else:
        dataset_class = get_dataset(config["dataset"])
        ds = dataset_class()
        grid_dir = os.path.join(GRID_ROOT, config["dataset"], out_dir)
    os.makedirs(grid_dir, exist_ok=True)
    with open(os.path.join(grid_dir, 'config.json'), 'w') as f:
        json.dump(config, f, sort_keys=True)
    for i in range(max_threads):
        thread = threading.Thread(target=training_thread, args=(i % len(devices), ds, config))
        thread.setDaemon(True)
        thread.start()
    queue.join()
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("Grid search complete!")
