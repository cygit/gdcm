# Guided Diverse Concept Miner (GDCM)
"Guided Diverse Concept Miner (GDCM): Uncovering Relevant Constructs for Managerial Insights from Text"

Find the paper here: https://pubsonline.informs.org/doi/10.1287/isre.2020.0494

End-to-end deep learning framework for automatic mining of diverse concepts guided by predicting the outcome variable.

## Prerequisites

   * Get the code: `git clone git@github.com:cygit/gdcm.git; cd gdcm`
   * The first time you install GDCM CLI, you must create a Python virtual environment and install the required packages. This may take several minutes.
```bash
python3 -m venv env
source env/bin/activate
cd src
pip install --editable .
```

From hereafter, each time you start a new command line instance, you must navigate to the `gdcm` directory as above and execute the commands:

```bash 
source env/bin/activate
cd src
```

## Quickstart
   * View help info `gdcm --help`
   * Train with 20 News Group dataset with default hyperparameters and save the outputs in folder "run0":
    `gdcm train news_group run0`. When this command starts running, logs will be saved in `run0/gdcm.log`.
   Metrics from each epoch will be written in `run0/train_metrics.txt`. Concept words will be saved in files
    `run0/concept/epoch*.txt` for each epoch. The state of the model will be saved every 10 epochs and at the last
     epoch in `run0/model/epoch*.pytorch`. A visualization of the concepts will be stored in `run0/visualization.html`.
   More options of this command can be seen with `gdcm train --help`. 
   * Start grid search with an example hyperparameter search space configuration for 20 News Group dataset: 
   `gdcm grid-search ../configs/news_config.json`. The grid search results will be saved in `../grid_search/news/run0`
   according to the `"out_dir"` specified in `gdcm/configs/news_config.json`. Under `../grid_search/news/run0`, 
   there will be directories named with the hash value of each set of the hyperparameters being searched, such as 
   `../grid_search/news/run0/0f735f978246aa65aa1806299869978c`. Within each of these directories, there are also log
   file `gdcm.log`, metrics file `train_metrics.txt`, concept words `concept/epoch*.txt`, and saved models 
   `model/epoch*.pytorch`. The best metrics of each set of hyperparameters done with training are stored in
    `../grid_search/news/run0/results.csv`.

## Usage
   * Usage of `gdcm train`:
   ```text
  Train GDCM

  DATASET is the name of the dataset to be used. It must be one of the
  datasets defined in `gdcm/src/dataset/` which is a subclass of BaseDataset.

  OUT-DIR is the path to the output directory where the model, results, and
  visualization will be saved

Options:
  --csv-path FILE                 Path to the csv file. Only needed if
                                  `dataset` is 'csv'

  --csv-text TEXT                 Column name of the text field in the csv
                                  file. Only needed if `dataset` is 'csv'

  --csv-label TEXT                Column name of the label field in the csv
                                  file. Only needed if `dataset` is 'csv'

  --nconcepts INTEGER             No. of concepts (default: 5)
  --embed-dim INTEGER             The size of each word/concept embedding
                                  vector (default: 50)

  --vocab-size INTEGER            Maximum vocabulary size (default: 5000)
  --nnegs INTEGER                 No. of negative samples (default: 5)
  --lam INTEGER                   Dirichlet loss weight (default: 10)
  --rho INTEGER                   Prediction loss weight (default: 100)
  --eta INTEGER                   Diversity loss weight (default: 10)
  --window-size INTEGER           Word embedding context window size (default:
                                  4)

  --lr FLOAT                      Learning rate (default: 0.01)
  --batch INTEGER                 Batch size (default: 100)
  --gpu INTEGER                   GPU device if CUDA is available. Ignored if
                                  no CUDA. (default: 0)

  --inductive TEXT                Whether to use inductive mode (default:
                                  True)

  --dropout FLOAT                 dropout rate applied on word/concept
                                  embedding (default: 0.0)

  --nepochs INTEGER               No. of epochs (default: 10)
  --concept-dist                  Concept vectors distance metric. Choices: ['dot'|'correlation'|'cosine'|'euclidean'|'hamming'] (default: 'dot')
  -h, --help                      Show this message and exit.
```
   * Usage of `gdcm grid-search`:
```
Usage: gdcm grid-search [OPTIONS] CONFIG

  Perform grid search with the given configuration

  CONFIG is the path of the config file

Options:
  -h, --help  Show this message and exit.
```
   * The configuration file for grid search is required. All of the possible fields of the configuration are 
   listed below with comments. Notice that values in `"dataset_params"`, `"gdcm_params"`, and `"fit_params"` must be 
   lists. During grid search, every possible combination of the values in these lists will be tried. The original 
   configuration file can be found at `gdcm/configs/news_config.json`.
```
{
      "dataset": "news_group", # the name of the dataset to be used. It must be one of the datasets 
                               # defined in `gdcm/src/dataset/` which is a subclass of BaseDataset. 
                               # The existing options are "news_group", "wcai", and "prosper_loan".
      "csv-path" : "/home/Downloads/data/wcai/wcai.csv", # path to the csv file. Only needed if `dataset` is 'csv'
      "csv-text" : "docs", # column name of the text field in the csv file. Only needed if `dataset` is 'csv'
      "csv-label" : "labels", # column name of the label field in the csv file. Only needed if `dataset` is 'csv'
      "gpus": [0],  # list of GPU devices if CUDA is available. Ignored if no CUDA.
      "max_threads": 1, # max number of parallel threads to run grid search
      "out_dir": "../grid_search/news/run0", # the directory to save grid search output files
      "dataset_params": { # hyperparameters for loading 20 News Group dataset
        "window_size": [4], # context window size for training word embedding
        "min_df": [0.01], # min document frequency of vocabulary
        "max_df": [0.8] # max document frequency of vocabulary
      },
      "gdcm_params": { # hyperparameters for creating GuidedDiverseConceptMiner instances
        "embed_dim": [50], # the size of each word/concept embedding vector
        "nnegs": [15], # the number of negative context words to be sampled during the training of word embeddings
        "nconcepts": [2], # the number of concepts
        "lam": [10, 100], # Dirichlet loss weight. The higher, the more sparse is the concept distribution of each document 
        "rho": [100, 1000], # Prediction loss weight. The higher, the more does the model focus on Prediction accuracy
        "eta": [100, 1000], # Diversity loss weight. The higher, the more different are the concept vectors from each other
        "inductive": [true], # whether to use neural network to inductively predict the concept weights of each document,
                             # or use a concept weights embedding
        "inductive_dropout": [0.01], # the dropout rate of the inductive neural network
        "hidden_size": [100], # the size of the hidden layers in the inductive neural network
        "num_layers": [1] # he number of layers in the inductive neural network
      },
      "fit_params": { # hyperparameters for training the GuidedDiverseConceptMiner
        "lr": [0.01, 0.001], # learning rate
        "nepochs": [30], # the number of training epochs
        "pred_only_epochs": [15], # the number of epochs optimized with prediction loss only
        "batch_size": [1024, 10000], # batch size
        "grad_clip": [1024] # maximum gradients magnitude. Gradients will be clipped within the range [-grad_clip, grad_clip]
      }
}
```
   * You can also choose to implement your own dataset under `gdcm/dataset`. It must be a subclass of 
   `dataset.base_dataset.BaseDataset` and implement the `load_data` method as other datasets `gdcm/dataset/wcai.py`, 
   `gdcm/dataset/news_group.py`, and `gdcm/dataset/prosper_loan.py`. The `load_data` is required to return 
   a dictionary containing the following attributes of the dataset:
   ```
{
    "bow_train": ndarray, shape (n_train_docs, vocab_size)
        Training corpus encoded as a bag-of-words matrix, where n_train_docs is the number of documents
        in the training set, and vocab_size is the vocabulary size.
    "y_train": ndarray, shape (n_train_docs,)
        Labels in the training set, ndarray with binary, multiclass, or continuous values.
    "bow_test": ndarray, shape (n_test_docs, vocab_size)
        Test corpus encoded as a matrix
    "y_test": ndarray, shape (n_test_docs,)
        Labels in the test set, ndarray with binary, multiclass, or continuous values.
    "doc_windows": ndarray, shape (n_windows, windows_size + 3)
        Context windows constructed from bow_train. Each row represents a context window, consisting of
        the document index of the context window, the encoded target words, the encoded context words,
        and the document's label. This can be generated with the helper function `get_windows`.
    "vocab" : array-like, shape `vocab_size`
        List of all the unique words in the training corpus. The order of this list corresponds
        to the columns of the `bow_train` and `bow_test`
    "word_counts": ndarray, shape (vocab_size,)
        The count of each word in the training documents. The ordering of these counts
        should correspond with `vocab`.
    "doc_lens" : ndarray, shape (n_train_docs,)
        The length of each training document.
    "expvars_train" [OPTIONAL] : ndarray, shape (n_train_docs, n_features)
        Extra features for making prediction during the training phase
    "expvars_test" [OPTIONAL] : ndarray, shape (n_test_docs, n_features)
        Extra features for making prediction during the testing phase
}
```
