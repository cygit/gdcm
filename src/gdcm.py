import logging
import os

import numpy as np
import pyLDAvis
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from torch.nn import Parameter
from sklearn.metrics import roc_auc_score, mean_squared_error

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import consts
from toolbox.alias_multinomial import AliasMultinomial

import optuna

torch.manual_seed(consts.SEED)
np.random.seed(consts.SEED)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class GuidedDiverseConceptMiner(nn.Module):

    def __init__(self, out_dir, embed_dim=300, nnegs=15, nconcepts=25, lam=100.0, rho=100.0, eta=1.0,
                 doc_concept_probs=None, word_vectors=None, theta=None, gpu=None,
                 inductive=True, inductive_dropout=0.01, hidden_size=100, num_layers=1,
                 bow_train=None, y_train=None, bow_test=None, y_test=None, doc_windows=None, vocab=None,
                 word_counts=None, doc_lens=None, expvars_train=None, expvars_test=None, file_log=False, norm=None):
        """A class representing a Focused Concept Miner which can mine concepts from unstructured text data while
        making high accuracy predictions with the mined concepts and optional structured data.

        Parameters
        ----------
        out_dir : str
            The directory to save output files from this instance
        embed_dim : int
            The size of each word/concept embedding vector
        nnegs : int
            The number of negative context words to be sampled during the training of word embeddings
        nconcepts : int
            The number of concepts
        lam : float
            Dirichlet loss weight. The higher, the more sparse is the concept distribution of each document
        rho : float
            Prediction loss weight. The higher, the more does the model focus on prediction accuracy
        eta : float
            Diversity loss weight. The higher, the more different are the concept vectors from each other
        doc_concept_probs [OPTIONAL] : ndarray, shape (n_train_docs, n_concepts)
            Pretrained concept distribution of each training document
        word_vectors [OPTIONAL] : ndarray, shape (vocab_size, embed_dim)
            Pretrained word embedding vectors
        theta [OPTIONAL] : ndarray, shape (n_concepts + 1) if `expvars_train` and `expvars_test` are None,
                            or (n_concepts + n_features + 1) `expvars_train` and `expvars_test` are not None
            Pretrained linear prediction weights
        gpu [OPTIONAL] : int
            CUDA device if CUDA is available
        inductive : bool
            Whether to use neural network to inductively predict the concept weights of each document,
            or use a concept weights embedding
        inductive_dropout : float
            The dropout rate of the inductive neural network
        hidden_size : int
            The size of the hidden layers in the inductive neural network
        num_layers : int
            The number of layers in the inductive neural network
        bow_train : ndarray, shape (n_train_docs, vocab_size)
            Training corpus encoded as a bag-of-words matrix, where n_train_docs is the number of documents
            in the training set, and vocab_size is the vocabulary size.
        y_train : ndarray, shape (n_train_docs,)
            Labels in the training set, ndarray with binary, multiclass, or continuos values.
        bow_test : ndarray, shape (n_test_docs, vocab_size)
            Test corpus encoded as a matrix
        y_test : ndarray, shape (n_test_docs,)
            Labels in the test set, ndarray with binary, multiclass, or continuos values.
        doc_windows : ndarray, shape (n_windows, windows_size + 3)
            Context windows constructed from `bow_train`. Each row represents a context window, consisting of
            the document index of the context window, the encoded target words, the encoded context words,
            and the document's label.
        vocab : array-like, shape `vocab_size`
            List of all the unique words in the training corpus. The order of this list corresponds
            to the columns of the `bow_train` and `bow_test`
        word_counts : ndarray, shape (vocab_size,)
            The count of each word in the training documents. The ordering of these counts
            should correspond with `vocab`.
        doc_lens : ndarray, shape (n_train_docs,)
            The length of each training document.
        expvars_train [OPTIONAL] : ndarray, shape (n_train_docs, n_features)
            Extra features for making prediction during the training phase
        expvars_test [OPTIONAL] : ndarray, shape (n_test_docs, n_features)
            Extra features for making prediction during the testing phase
        file_log : bool
            Whether writes logs into a file or stdout
        norm : string
            Normalization method to apply on the label data (Y variable) if they are continuous (default: 'standard')
        """
        super(GuidedDiverseConceptMiner, self).__init__()
        ndocs = bow_train.shape[0]
        vocab_size = bow_train.shape[1]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.concept_dir = os.path.join(out_dir, "concept")
        os.makedirs(self.concept_dir, exist_ok=True)
        self.model_dir = os.path.join(out_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.embed_dim = embed_dim
        self.nnegs = nnegs
        self.nconcepts = nconcepts
        self.lam = lam
        self.rho = rho
        self.eta = eta
        self.alpha = 1.0 / nconcepts
        self.expvars_train = expvars_train
        self.expvars_test = expvars_test
        # print(self.expvars_train[0])
        self.inductive = inductive
        if torch.cuda.is_available():
                self.device = "cuda" 
        elif torch.backends.mps.is_available():
           self.device = "mps"
        else:
            self.device = 'cpu'
           
        
        device = torch.device(self.device)
        self.bow_train = torch.tensor(bow_train, dtype=torch.float32, requires_grad=False, device=device)
        assert not (self.inductive and self.bow_train is None)
        self.y_train = y_train
        self.bow_test = torch.tensor(bow_test, dtype=torch.float32, requires_grad=False, device=device)
        self.y_test = y_test

        """code optimization task 1 (normalization)"""
        #self.validate_labels() #checking if the labels are binary or continuous.

        if norm is not None:
            self.norm = norm
            
            if not self.check_binary_labels(y_train) and not self.check_multiclass_labels(y_train): #y_train is continuous
                self.y_train = self.normalize_labels(self.y_train, self.norm)
                print(f" continuous data, {self.norm} normalized ")
                print(self.y_train)

            if not self.check_binary_labels(y_test) and not self.check_multiclass_labels(y_test):
                self.y_test = self.normalize_labels(self.y_test, self.norm)
                print(f" continuous data, {self.norm} normalized ")
                print(self.y_test)

        #print(self.y_train)
        self.is_binary = self.check_binary_labels(y_train) and self.check_binary_labels(y_test)

        self.is_multiclass = self.check_multiclass_labels(y_train) or self.check_multiclass_labels(y_test)
        
        if self.is_multiclass:
            self.num_classes = max(len(np.unique(y_train)),len(np.unique(y_test)))
        
        
        
        
        
        
        self.train_dataset = DocWindowsDataset(doc_windows)

        if doc_lens is None:
            self.docweights = np.ones(ndocs, dtype=np.float)
        else:
            self.docweights = 1.0 / np.log(doc_lens)
            self.doc_lens = doc_lens
        self.docweights = torch.tensor(self.docweights, dtype=torch.float32, requires_grad=False, device=device)

        if expvars_train is not None:
            self.expvars_train = torch.tensor(expvars_train, dtype=torch.float32, requires_grad=False, device=device)
        if expvars_test is not None:
            self.expvars_test = torch.tensor(expvars_test, dtype=torch.float32, requires_grad=False, device=device)
        # word embedding
        self.embedding_i = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim,
                                        sparse=False)
        if word_vectors is not None:
            self.embedding_i.weight.data = torch.FloatTensor(word_vectors)
        else:
            torch.nn.init.kaiming_normal_(self.embedding_i.weight)

        # regular embedding for concepts (never indexed so not sparse)
        self.embedding_t = nn.Parameter(torch.FloatTensor(ortho_group.rvs(embed_dim)[0:nconcepts]))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if file_log:
            log_path = os.path.join(out_dir, "gdcm.log")
            print("Saving logs in the file " + os.path.abspath(log_path))
            logging.basicConfig(filename=log_path,
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # embedding for per-document concept weights
        if self.inductive:
            weight_generator_network = []
            if num_layers > 0:
                # input layer
                weight_generator_network.extend([torch.nn.Linear(vocab_size, hidden_size),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Dropout(inductive_dropout)])
                # hidden layers
                for h in range(num_layers):
                    weight_generator_network.extend([torch.nn.Linear(hidden_size, hidden_size),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(inductive_dropout)])
                # output layer
                weight_generator_network.append(torch.nn.Linear(hidden_size,
                                                                nconcepts))
            else:
                weight_generator_network.append(torch.nn.Linear(vocab_size,
                                                                nconcepts))
            for m in weight_generator_network:
                if type(m) == torch.nn.Linear:
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.normal_(m.bias)
            self.doc_concept_network = torch.nn.Sequential(*weight_generator_network)
        else:
            self.doc_concept_weights = nn.Embedding(num_embeddings=ndocs,
                                                    embedding_dim=nconcepts,
                                                    sparse=False)
            if doc_concept_probs is not None:
                self.doc_concept_weights.weight.data = torch.FloatTensor(doc_concept_probs)
            else:
                torch.nn.init.kaiming_normal_(self.doc_concept_weights.weight)

        if theta is not None:
            self.theta = Parameter(torch.FloatTensor(theta))
        # explanatory variables
        else:
            if expvars_train is not None:
                # TODO: add assert shape
                nexpvars = expvars_train.shape[1]
                if self.is_multiclass:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1))  # for binary and continuous variables
            
            
            else:
                if self.is_multiclass:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1))  # for binary and continuous variables
        
        torch.nn.init.normal_(self.theta)

        # enable gradients (True by default, just confirming)
        self.embedding_i.weight.requires_grad = True
        self.embedding_t.requires_grad = True
        self.theta.requires_grad = True

        # weights for negative sampling
        wf = np.power(word_counts, consts.BETA)  # exponent from word2vec paper
        self.word_counts = word_counts
        wf = wf / np.sum(wf)  # convert to probabilities
        self.weights = torch.tensor(wf, dtype=torch.float32, requires_grad=False, device=device)
        self.vocab = vocab
        # dropout
        self.dropout1 = nn.Dropout(consts.PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(consts.DOC_VECS_DROPOUT)
        self.multinomial = AliasMultinomial(wf, self.device)

    """ code optimization task 1 (for checking if y_train/y_test is binary or continuous in the existing datasets)"""

    def check_binary_labels(self, y):
        unique_values = np.unique(y)
        
        return (len(unique_values) == 2 and set(unique_values).issubset({0, 1}))

    
    def check_multiclass_labels(self, y):
        unique_values = np.unique(y)
        num_of_vals = len(unique_values)
        return num_of_vals > 2 and not np.issubdtype(y.dtype, np.number)


    def validate_labels(self):
        if self.check_binary_labels(self.y_train):
            print("y_train is binary")
        else:
            print("y_train is continuous, normalizing y_train...")
        if self.check_binary_labels(self.y_test):
            print("y_test is binary")
        else:
            print("y_test is continuous, normalizing y_test...")
    
    def normalize_labels(self, data, method='standard'):
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported normalization method")
        
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()




    def forward(self, doc, target, contexts, labels, per_doc_loss=None):
        """
        Args:
            doc:        [batch_size,1] LongTensor of document indices
            target:     [batch_size,1] LongTensor of target (pivot) word indices
            contexts:   [batchsize,window_size] LongTensor of convext word indices
            labels:     [batchsize,1] LongTensor of document labels

            All arguments are tensors wrapped in Variables.
        """
        batch_size, window_size = contexts.size()

        # reweight loss by document length
        w = autograd.Variable(self.docweights[doc.data]).to(self.device)
        w /= w.sum()
        w *= w.size(0)

        # construct document vector = weighted linear combination of concept vectors
        if self.inductive:
            doc_concept_weights = self.doc_concept_network(self.bow_train[doc])
        else:
            doc_concept_weights = self.doc_concept_weights(doc)
        doc_concept_probs = F.softmax(doc_concept_weights, dim=1)
        doc_concept_probs = doc_concept_probs.unsqueeze(1)  # (batches, 1, T)
        concept_embeddings = self.embedding_t.expand(batch_size, -1, -1)  # (batches, T, E)
        doc_vector = torch.bmm(doc_concept_probs, concept_embeddings)  # (batches, 1, E)
        doc_vector = doc_vector.squeeze(dim=1)  # (batches, E)
        doc_vector = self.dropout2(doc_vector)

        # sample negative word indices for negative sampling loss; approximation by sampling from the whole vocab
        if self.device == "cpu":
            nwords = torch.multinomial(self.weights, batch_size * window_size * self.nnegs,
                                       replacement=True).view(batch_size, -1)
            nwords = autograd.Variable(nwords)
        else:
            nwords = self.multinomial.draw(batch_size * window_size * self.nnegs)
            nwords = autograd.Variable(nwords).view(batch_size, window_size * self.nnegs)

        # compute word vectors
        ivectors = self.dropout1(self.embedding_i(target))  # (batches, E)
        ovectors = self.embedding_i(contexts)  # (batches, window_size, E)
        nvectors = self.embedding_i(nwords).neg()  # row vector

        # construct "context" vector defined by lda2vec
        context_vectors = doc_vector + ivectors
        context_vectors = context_vectors.unsqueeze(2)  # (batches, E, 1)

        # compose negative sampling loss
        oloss = torch.bmm(ovectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=consts.EPS).log().sum(1)
        nloss = torch.bmm(nvectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=consts.EPS).log().sum(1)
        negative_sampling_loss = (oloss + nloss).neg()
        negative_sampling_loss *= w  # downweight loss for each document
        negative_sampling_loss = negative_sampling_loss.mean()  # mean over the batch

        # compose dirichlet loss
        doc_concept_probs = doc_concept_probs.squeeze(dim=1)  # (batches, T)
        doc_concept_probs = doc_concept_probs.clamp(min=consts.EPS)
        dirichlet_loss = doc_concept_probs.log().sum(1)  # (batches, 1)
        dirichlet_loss *= self.lam * (1.0 - self.alpha)
        dirichlet_loss *= w  # downweight loss for each document
        dirichlet_loss = dirichlet_loss.mean()  # mean over the entire batch

        ones = torch.ones((batch_size, 1)).to(self.device)
        doc_concept_probs = torch.cat((ones, doc_concept_probs), dim=1)

        # expand doc_concept_probs vector with explanatory variables
        if self.expvars_train is not None:
            doc_concept_probs = torch.cat((doc_concept_probs, self.expvars_train[doc, :]),
                                          dim=1)
        # compose prediction loss
        # [batch_size] = torch.matmul([batch_size, nconcepts], [nconcepts])
        # pred_weight = torch.matmul(doc_concept_probs.unsqueeze(0), self.theta).squeeze(0)
        # print(doc_concept_probs.shape)
        # print(self.theta.shape)
        pred_weight = torch.matmul(doc_concept_probs, self.theta)
        # print(pred_weight)
        # print(labels)
   
        if self.is_binary:
            pred_loss = F.binary_cross_entropy_with_logits(pred_weight, labels,
                                                       weight=w, reduction='none')
        elif self.is_multiclass:
            pred_loss = F.cross_entropy(pred_weight, labels.long(), reduction = 'none')
        else:
            pred_loss = F.mse_loss(pred_weight, labels, reduction='none')
            pred_loss = pred_loss * w # applying the weight element-wise to the calcuated loss

        pred_loss *= self.rho
        pred_loss = pred_loss.mean()

        # compose diversity loss
        #   1. First compute \sum_i \sum_j log(sigmoid(T_i, T_j))
        #   2. Then compute \sum_i log(sigmoid(T_i, T_i))
        #   3. Loss = (\sum_i \sum_j log(sigmoid(T_i, T_j)) - \sum_i log(sigmoid(T_i, T_i)) )
        #           = \sum_i \sum_{j > i} log(sigmoid(T_i, T_j))
        div_loss = torch.mm(self.embedding_t,
                            torch.t(self.embedding_t)).sigmoid().clamp(min=consts.EPS).log().sum() \
                   - (self.embedding_t * self.embedding_t).sigmoid().clamp(min=consts.EPS).log().sum()
        div_loss /= 2.0  # taking care of duplicate pairs T_i, T_j and T_j, T_i
        div_loss = div_loss.repeat(batch_size)
        div_loss *= w  # downweight by document lengths
        div_loss *= self.eta
        div_loss = div_loss.mean()  # mean over the entire batch

        return negative_sampling_loss, dirichlet_loss, pred_loss, div_loss

    def fit(self, lr=0.01, nepochs=200, pred_only_epochs=20,
            batch_size=100, weight_decay=0.01, grad_clip=5, save_epochs=10, concept_dist="dot"):
        """
        Train the GDCM model

        Parameters
        ----------
        lr : float
            Learning rate
        nepochs : int
            The number of training epochs
        pred_only_epochs : int
            The number of epochs optimized with prediction loss only
        batch_size : int
            Batch size
        weight_decay : float
            Adam optimizer weight decay (L2 penalty)
        grad_clip : float
            Maximum gradients magnitude. Gradients will be clipped within the range [-grad_clip, grad_clip]
        save_epochs : int
            The number of epochs in between saving the model weights
        concept_dist: str
            Concept vectors distance metric. Choices are 'dot', 'correlation', 'cosine', 'euclidean', 'hamming'.

        Returns
        -------
        metrics : ndarray, shape (n_epochs, 6)
            Training metrics from each epoch including: total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss,
            avg_diversity_loss, train_auc, test_auc
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=True,
                                                       num_workers=4, pin_memory=True,
                                                       drop_last=False)

        
       
        self.to(self.device)

        train_metrics_file = open(os.path.join(self.out_dir, "train_metrics.txt"), "w")
        train_metrics_file.write("total_loss,avg_sgns_loss,avg_dirichlet_loss,avg_pred_loss,"
                                 "avg_div_loss,train_auc,test_auc\n")

        # SGD generalizes better: https://arxiv.org/abs/1705.08292
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        nwindows = len(self.train_dataset)
        results = []
        for epoch in range(nepochs):
            total_sgns_loss = 0.0
            total_dirichlet_loss = 0.0
            total_pred_loss = 0.0
            total_diversity_loss = 0.0

            self.train()
            for batch in train_dataloader:
                batch = batch.long()
                batch = batch.to(self.device)
                doc = batch[:, 0]
                iword = batch[:, 1]
                owords = batch[:, 2:-1]
                labels = batch[:, -1].float()

                sgns_loss, dirichlet_loss, pred_loss, div_loss = self(doc, iword, owords, labels)
                if epoch < pred_only_epochs:
                    loss = pred_loss
                else:
                    loss = sgns_loss + dirichlet_loss + pred_loss + div_loss
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                for p in self.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

                nsamples = batch.size(0)

                total_sgns_loss += sgns_loss.detach().cpu().numpy() * nsamples
                total_dirichlet_loss += dirichlet_loss.detach().cpu().numpy() * nsamples
                total_pred_loss += pred_loss.data.detach().cpu().numpy() * nsamples
                total_diversity_loss += div_loss.data.detach().cpu().numpy() * nsamples

            train_auc = self.calculate_auc("Train", self.bow_train, self.y_train, self.expvars_train)
            test_auc = 0.0
            if self.inductive:
                test_auc = self.calculate_auc("Test", self.bow_test, self.y_test, self.expvars_test)

            total_loss = (total_sgns_loss + total_dirichlet_loss + total_pred_loss + total_diversity_loss) / nwindows
            avg_sgns_loss = total_sgns_loss / nwindows
            avg_dirichlet_loss = total_dirichlet_loss / nwindows
            avg_pred_loss = total_pred_loss / nwindows
            avg_diversity_loss = total_diversity_loss / nwindows
            self.logger.info("epoch %d/%d:" % (epoch, nepochs))
            self.logger.info("Total loss: %.4f" % total_loss)
            self.logger.info("SGNS loss: %.4f" % avg_sgns_loss)
            self.logger.info("Dirichlet loss: %.4f" % avg_dirichlet_loss)
            self.logger.info("Prediction loss: %.4f" % avg_pred_loss)
            self.logger.info("Diversity loss: %.4f" % avg_diversity_loss)
            concepts = self.get_concept_words(concept_dist=concept_dist)
            with open(os.path.join(self.concept_dir, "epoch%d.txt" % epoch), "w") as concept_file:
                for i, concept_words in enumerate(concepts):
                    self.logger.info('concept %d: %s' % (i + 1, ' '.join(concept_words)))
                    concept_file.write('concept %d: %s\n' % (i + 1, ' '.join(concept_words)))
            metrics = (total_loss, avg_sgns_loss, avg_dirichlet_loss, avg_pred_loss,
                       avg_diversity_loss, train_auc, test_auc)
            train_metrics_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % metrics)
            train_metrics_file.flush()
            results.append(metrics)
            if (epoch + 1) % save_epochs == 0:
                torch.save(self.state_dict(), os.path.join(self.model_dir, "epoch%d.pytorch" % epoch))
                with torch.no_grad():
                    doc_concept_probs = self.get_train_doc_concept_probs()
                    np.save(os.path.join(self.model_dir, "epoch%d_train_doc_concept_probs.npy" % epoch),
                            doc_concept_probs.cpu().detach().numpy())

        torch.save(self.state_dict(), os.path.join(self.model_dir, "epoch%d.pytorch" % (nepochs - 1)))
        return np.array(results)

    def calculate_auc(self, split, X, y, expvars):
        y_pred = self.predict_proba(X, expvars).cpu().detach().numpy()
        if self.is_binary:
            auc = roc_auc_score(y, y_pred)
            self.logger.info("%s AUC: %.4f" % (split, auc))
            return auc
        elif self.is_multiclass:
            y = np.asarray(y).astype(int)
        
            num_classes = len(np.unique(y))
            
            if num_classes > 1:  
                
                y_pred_normalized = y_pred / y_pred.sum(axis=1, keepdims=True)
                auc = roc_auc_score(y, y_pred_normalized, multi_class='ovr')
                self.logger.info("%s AUC (OvR): %.4f" % (split, auc))
                return auc
            else:
                #in case only one class is present
                self.logger.warning("Only one class present in true labels. ROC AUC score is not defined in that case.")
                return None  
        else:
            mse = mean_squared_error(y, y_pred)
            # mae = mean_absolute_error(y, y_pred)
            # r2 = r2_score(y, y_pred)
            self.logger.info("%s MSE: %.4f" % (split, mse))
            return mse
            

    def predict_proba(self, count_matrix, expvars=None):
        with torch.no_grad():
            batch_size = count_matrix.size(0)
            if self.inductive:
                doc_concept_weights = self.doc_concept_network(count_matrix)
            else:
                doc_concept_weights = self.doc_concept_weights.weight.data
            doc_concept_probs = F.softmax(doc_concept_weights, dim=1)  # convert to probabilities
            ones = torch.ones((batch_size, 1)).to(self.device)
            doc_concept_probs = torch.cat((ones, doc_concept_probs), dim=1)

            if expvars is not None:
                doc_concept_probs = torch.cat((doc_concept_probs, expvars), dim=1)

            pred_weight = torch.matmul(doc_concept_probs, self.theta)
            pred_proba = pred_weight.sigmoid()
        return pred_proba

    def get_train_doc_concept_probs(self):
        if self.inductive:
            doc_concept_weights = self.doc_concept_network(self.bow_train)
        else:
            doc_concept_weights = self.doc_concept_weights.weight.data
        return F.softmax(doc_concept_weights, dim=1)  # convert to probabilities

    def visualize(self):
        with torch.no_grad():
            doc_concept_probs = self.get_train_doc_concept_probs()
            # [n_concepts, vocab_size] weighted word counts of each concept
            concept_word_counts = torch.matmul(doc_concept_probs.transpose(0, 1), self.bow_train)
            # normalize word counts to word distribution of each concept
            concept_word_dists = concept_word_counts / concept_word_counts.sum(1, True)
            # fill NaN with 1/vocab_size in case a concept has all zero word distribution
            concept_word_dists[concept_word_dists != concept_word_dists] = 1.0 / concept_word_dists.shape[1]
            vis_data = pyLDAvis.prepare(topic_term_dists=concept_word_dists.data.cpu().numpy(),
                                        doc_topic_dists=doc_concept_probs.data.cpu().numpy(),
                                        doc_lengths=self.doc_lens, vocab=self.vocab, term_frequency=self.word_counts)
            
            html_path = os.path.join(self.out_dir, "visualization.html")
            pyLDAvis.save_html(vis_data, html_path)
            # pyLDAvis.save_html(vis_data, os.path.join(self.out_dir, "visualization.html"))

            for i in range(len(concept_word_dists)):
                concept_word_weights = dict(zip(self.vocab, concept_word_dists[i].cpu().numpy()))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(concept_word_weights)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Concept {i+1} Word Cloud')
                plt.axis('off')
                plt.savefig(os.path.join(self.out_dir, f'concept_{i+1}_wordcloud.png'))

            with open(html_path, "a+") as f:

                current_directory = os.getcwd()
                print("Current directory:", current_directory)
                swiper_vis_path = current_directory + '/swiper.html'
                with open(swiper_vis_path, 'r') as swipertext:
                    swiper = swipertext.read() 
                    
                f.write(swiper)
            

    # TODO: add filtering such as pos and tf
    def get_concept_words(self, top_k=10, concept_dist='dot'):
        concept_embed = self.embedding_t.data.cpu().numpy()
        word_embed = self.embedding_i.weight.data.cpu().numpy()
        if concept_dist == 'dot':
            dist = -np.matmul(concept_embed, np.transpose(word_embed, (1, 0)))
        else:
            dist = cdist(concept_embed, word_embed, metric=concept_dist)
        nearest_word_idxs = np.argsort(dist, axis=1)[:, :top_k]  # indices of words with min cosine distance
        concepts = []
        for j in range(self.nconcepts):
            nearest_words = [self.vocab[i] for i in nearest_word_idxs[j, :]]
            concepts.append(nearest_words)
        return concepts


class DocWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.data = windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
