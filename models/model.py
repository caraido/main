# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import  Dataset
from multiprocessing import Pool
from utils.utils import reformat
from utils.retrieval import mean_embedding_per_word, mean_center_db
from sklearn.model_selection import train_test_split


def _worker_init():
    """Pool worker initialiser: suppress all warnings in subprocesses."""
    import warnings as _w
    _w.filterwarnings('ignore')


# Define PyTorch model, with dropout at hidden layers
class BottleneckModel(nn.Module):
    def __init__(self,n_features, n_classes, n_bottlenecks=5, dropout_rate=0.1):
        # n_features is the concatenated channels and time information
        # n_classes is the number of labels,
        super().__init__()
        self.linear = nn.Linear(n_features, n_bottlenecks)
        self.batch_norm = nn.BatchNorm1d(n_bottlenecks)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(n_bottlenecks, n_classes)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class BasicClassifier:
    def __init__(self,extractor, decoder,scaler=None):
        self.extractor=extractor
        self.decoder=decoder
        self.scaler=scaler

        self.n_epochs=0
        self.n_bins_history=0
        self.split=0

        self.labels=None
        self.data=None
        self.X_to_use=None
        self.n_bins=None

        self.all_test_accuracy=[]
        self.all_chance=[]
        self.all_train_accuracy=[]

        self.all_predicted_labels=[]
        self.all_true_labels=[]

    def load_data(self, data, labels, split=0.3, n_bins_history=10):
        data=np.squeeze(np.array(data)[labels!='NA'])
        # this is for single channel data
        if data.ndim==2:
            data=data[:,:,np.newaxis]
        labels=labels[labels!='NA']

        self.data=data
        self.labels=labels
        self.split=split
        self.n_bins_history=n_bins_history

        self.X_to_use=reformat(self.data,
                               self.n_bins_history)
        self.n_bins=len(self.X_to_use)

    def fit(self, n_epochs=50, parallel=None, use_kfold=False, n_splits=5):
        """
        Fit the classifier.
        
        Parameters:
        -----------
        n_epochs : int
            Number of random repeats/epochs
        parallel : int or None
            Number of parallel processes. If None, use sequential processing (no multiprocessing)
        use_kfold : bool
            If True, use StratifiedKFold for each epoch; if False, use random train_test_split
        n_splits : int
            Number of folds for StratifiedKFold (only used if use_kfold=True)
        """
        self.n_epochs=n_epochs
        self.use_kfold=use_kfold
        self.n_splits=n_splits
        self.all_train_labels=[]
        self.all_predicted_labels=[]
        self.all_test_accuracy=[]
        self.all_train_accuracy=[]
        self.all_chance=[]

        if parallel is None:
            # Sequential processing
            results = [self._fit(i) for i in range(self.n_epochs)]
        else:
            # Parallel processing
            with Pool(processes=parallel, initializer=_worker_init) as pool:
                results = pool.map(self._fit,range(self.n_epochs))

        all_train_accuracy=[]
        all_test_accuracy=[]
        all_chance=[]

        all_true_labels=[]
        all_predicted_labels=[]

        for result in results:
            train_accuracy,test_accuracy,chance,true_labels,predicted_labels=result
            all_train_accuracy.append(train_accuracy)
            all_test_accuracy.append(test_accuracy)
            all_chance.append(chance)
            all_true_labels.append(true_labels)
            all_predicted_labels.append(predicted_labels)

        self.all_train_accuracy=np.array(all_train_accuracy)
        self.all_test_accuracy=np.array(all_test_accuracy)
        self.all_chance=np.array(all_chance)
        self.all_true_labels=np.array(all_true_labels)
        self.all_predicted_labels=np.array(all_predicted_labels)

    def _fit(self,_):
        if self.X_to_use:
            all_train_accuracy=[] 
            all_test_accuracy=[]
            all_chance=[]

            all_true_labels=[]
            all_test_labels=[]
            for n_bin in range(len(self.X_to_use)):
                X = self.X_to_use[n_bin]
                y = self.labels
                
                if self.use_kfold:
                    # Use StratifiedKFold
                    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=None)
                    fold_train_accs = []
                    fold_test_accs = []
                    fold_chances = []
                    fold_true_labels = []
                    fold_test_labels = []
                    
                    for train_idx, test_idx in skf.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # scale the data if the scaler is not None
                        if self.scaler is not None:
                            X_train=self.scaler.fit_transform(X_train)
                            X_test=self.scaler.transform(X_test)
                        
                        # extract the features
                        if self.extractor is not None:
                            self.extractor.fit(X_train,y_train)
                            X_train_low = self.extractor.transform(X_train)
                            X_test_low = self.extractor.transform(X_test)
                        else:
                            X_train_low=X_train
                            X_test_low=X_test
                        
                        # classification
                        self.decoder.fit(X_train_low, y_train)
                        y_test_predict = self.decoder.predict(X_test_low)
                        y_train_predict = self.decoder.predict(X_train_low)
                        
                        # get the chance
                        X_train_shuffle = np.random.permutation(X_train_low.flatten()).reshape(X_train_low.shape)
                        self.decoder.fit(X_train_shuffle, y_train)
                        y_shuffle = self.decoder.predict(X_test_low)
                        
                        # calculate the accuracy
                        correct_train = np.sum(y_train_predict == y_train)
                        accuracy_train = correct_train / len(y_train_predict)
                        correct_test = np.sum(y_test_predict == y_test)
                        accuracy_test = correct_test / len(y_test_predict)
                        
                        # calculate the chance
                        correct_shuffle = np.sum(y_shuffle == y_test)
                        chance = correct_shuffle / len(y_shuffle)
                        
                        fold_train_accs.append(accuracy_train)
                        fold_test_accs.append(accuracy_test)
                        fold_chances.append(chance)
                        fold_true_labels.append(y_test)
                        fold_test_labels.append(y_test_predict)
                    
                    # Average across folds
                    all_train_accuracy.append(np.mean(fold_train_accs))
                    all_test_accuracy.append(np.mean(fold_test_accs))
                    all_chance.append(np.mean(fold_chances))
                    # Concatenate labels from all folds
                    all_true_labels.append(np.concatenate(fold_true_labels))
                    all_test_labels.append(np.concatenate(fold_test_labels))
                    
                else:
                    # Use random train_test_split (original behavior)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split)
                    
                    # scale the data if the scaler is not None
                    if self.scaler is not None:
                        X_train=self.scaler.fit_transform(X_train)
                        X_test=self.scaler.transform(X_test)

                    # extract the features
                    if self.extractor is not None:
                        self.extractor.fit(X_train,y_train)
                        X_train_low = self.extractor.transform(X_train)
                        X_test_low = self.extractor.transform(X_test)
                    else:
                        X_train_low=X_train
                        X_test_low=X_test

                    # classification
                    self.decoder.fit(X_train_low, y_train)
                    y_test_predict = self.decoder.predict(X_test_low)
                    y_train_predict = self.decoder.predict(X_train_low)
                    # get the chance
                    X_train_shuffle = np.random.permutation(X_train_low.flatten()).reshape(X_train_low.shape)
                    self.decoder.fit(X_train_shuffle, y_train)
                    y_shuffle = self.decoder.predict(X_test_low)

                    # calculate the accuracy
                    correct_train = np.sum(y_train_predict == y_train)
                    accuracy_train = correct_train / len(y_train_predict)
                    correct_test = np.sum(y_test_predict == y_test)
                    accuracy_test = correct_test / len(y_test_predict)

                    # calculate the chance
                    correct_shuffle = np.sum(y_shuffle == y_test)
                    chance = correct_shuffle / len(y_shuffle)

                    all_train_accuracy.append(accuracy_train)
                    all_test_accuracy.append(accuracy_test)
                    all_chance.append(chance)

                    # save the true and predicted labels
                    all_true_labels.append(y_test)
                    all_test_labels.append(y_test_predict)

            return (np.array(all_train_accuracy),
                    np.array(all_test_accuracy),
                    np.array(all_chance),
                    np.array(all_true_labels,dtype=object),
                    np.array(all_test_labels,dtype=object))
        
    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.extractor is not None:
            X = self.extractor.transform(X)

        return self.decoder.predict(X)


class BasicRegressor:
    # the regressor can be CCA, PLS, RidgeRegressor, etc
    # We will add dimension reduction method on both sides or x and y but it could just be None

    def __init__(self,regressor,x_reducer=None, y_reducer=None):
        self.regressor=regressor
        self.x_reducer=x_reducer
        self.y_reducer=y_reducer

        self.n_epochs=0
        self.n_bins_history=0
        self.split=0

        self.y=None
        self.data=None
        self.X_to_use=None
        self.n_bins=None
        self._closest=None

        self.all_x_reducer_weights=None
        self.all_y_reducer_weights=None

        self.all_regressor_weights=[]
        self.all_test_score=[]
        self.all_train_score=[]
        self.all_cosine_sim=[]          # mean cosine similarity (pred vs true)
        self.all_train_cosine_sim=[]    # mean cosine similarity on train set
        self.all_chance=[]
        self.all_top_k_accuracy={}

        # Retrieval bookkeeping (memory-friendly index representation)
        self.word_to_index={}
        self.index_to_word=np.array([], dtype=object)
        self.category_to_index={}
        self.index_to_category=np.array([], dtype=object)
        self.word_index_to_category_index=None

        self._index_dtype=np.uint32
        self._retrieval_db_embeds_raw=None
        self._retrieval_db_word_idx=None

        self.save_retrieval_pairs=False
        self.include_label_strings=False
        self.all_retrieval_pairs=[]
        self.all_retrieval_top1=[]
        self.all_retrieval_top3=[]
        self.all_retrieval_top5=[]
        self.all_retrieval_chance_top1=[]
        self.all_retrieval_chance_top3=[]
        self.all_retrieval_chance_top5=[]
        self.all_retrieval_category_top1=[]
        self.all_retrieval_category_chance_top1=[]

        self.all_retrieval_word_balanced_acc=[]
        self.all_retrieval_chance_word_balanced_acc=[]
        self.all_retrieval_word_f1=[]
        self.all_retrieval_chance_word_f1=[]
        self.all_retrieval_category_balanced_acc=[]
        self.all_retrieval_category_chance_balanced_acc=[]
        self.all_retrieval_category_f1=[]
        self.all_retrieval_category_chance_f1=[]

    def load_data(self, data, y, split=0.3, n_bins_history=10, labels=None, category_labels=None):
        # here data is the neural activity
        # y is the word/picture/any embeddings
        # labels is an optional array of string labels (e.g. word identities) for retrieval
        # category_labels is an optional array mapping each sample label to a super-category
        self.data = data
        self.y = np.asarray(y)
        self.split = split
        self.n_bins_history = n_bins_history
        self.labels = np.asarray(labels) if labels is not None else None

        if self.labels is not None:
            unique_words, sample_word_idx = np.unique(self.labels, return_inverse=True)
            self.index_to_word = np.asarray(unique_words)
            self.word_to_index = {word: idx for idx, word in enumerate(self.index_to_word)}

            n_words = len(self.index_to_word)
            if n_words <= np.iinfo(np.uint16).max:
                self._index_dtype = np.uint16
            elif n_words <= np.iinfo(np.uint32).max:
                self._index_dtype = np.uint32
            else:
                self._index_dtype = np.uint64

            if category_labels is not None:
                category_labels = np.asarray(category_labels)
                if len(category_labels) != len(self.labels):
                    raise ValueError(
                        "category_labels must have the same length as labels"
                    )

                # Category labels are trial-wise and may come from target words,
                # while retrieval labels can come from answered words. If a single
                # answered word spans multiple target categories, assign the
                # category by majority vote to keep category metrics well-defined.
                category_votes = [dict() for _ in range(n_words)]
                for wi, cat in zip(sample_word_idx, category_labels):
                    votes = category_votes[wi]
                    votes[cat] = votes.get(cat, 0) + 1

                word_categories = np.empty(n_words, dtype=object)
                conflicts = []
                for wi, votes in enumerate(category_votes):
                    if not votes:
                        word_categories[wi] = 'unknown'
                        continue

                    ranked = sorted(votes.items(), key=lambda x: (-x[1], str(x[0])))
                    word_categories[wi] = ranked[0][0]
                    if len(ranked) > 1:
                        conflicts.append((self.index_to_word[wi], ranked, ranked[0][0]))

                if conflicts:
                    preview = '; '.join(
                        f"{w}: {r} -> {c}"
                        for w, r, c in conflicts[:5]
                    )
                    warnings.warn(
                        "Some answered-word labels map to multiple target-derived "
                        "categories. Using majority-vote category assignment per "
                        f"word. Examples: {preview}",
                        RuntimeWarning,
                    )

                unique_categories = np.unique(word_categories)
                self.index_to_category = np.asarray(unique_categories)
                self.category_to_index = {
                    cat: idx for idx, cat in enumerate(self.index_to_category)
                }
                self.word_index_to_category_index = np.array(
                    [self.category_to_index[cat] for cat in word_categories],
                    dtype=np.int32,
                )
            else:
                self.category_to_index = {}
                self.index_to_category = np.array([], dtype=object)
                self.word_index_to_category_index = None
        else:
            if category_labels is not None:
                raise ValueError("category_labels requires labels to be provided")
            self.word_to_index = {}
            self.index_to_word = np.array([], dtype=object)
            self.category_to_index = {}
            self.index_to_category = np.array([], dtype=object)
            self.word_index_to_category_index = None

        self.X_to_use=reformat(self.data,
                               self.n_bins_history)
        self.n_bins = len(self.X_to_use)

    def fit(self, n_epochs=50, parallel=None, closest='l2', use_kfold=False, n_splits=5, 
            compute_top_k_accuracy=True, top_k_values=[1, 3, 5, 10],
            compute_retrieval=False, save_retrieval_pairs=False,
            include_label_strings=False):
        """
        Fit the regressor.
        
        Parameters:
        -----------
        n_epochs : int
            Number of random repeats/epochs
        parallel : int or None
            Number of parallel processes. If None, use sequential processing (no multiprocessing)
        closest : str
            Method to find closest predictions ('l2' or 'l1')
        use_kfold : bool
            If True, use KFold for each epoch; if False, use random train_test_split
        n_splits : int
            Number of folds for KFold (only used if use_kfold=True)
        compute_top_k_accuracy : bool
            If True, compute top-k accuracy for predictions
        top_k_values : list
            List of k values for top-k accuracy computation
        compute_retrieval : bool
            If True and labels were provided to load_data, compute retrieval
            accuracy (top-1) at each time bin.
        save_retrieval_pairs : bool
            If True, store compact true/predicted word index pairs for each test split.
        include_label_strings : bool
            If True and save_retrieval_pairs is enabled, also store decoded word strings
            alongside indices (larger memory footprint).
        """
        self.n_epochs=n_epochs
        self._closest=closest
        self.use_kfold=use_kfold
        self.n_splits=n_splits
        self.compute_top_k_accuracy=compute_top_k_accuracy
        self.top_k_values=top_k_values
        self.compute_retrieval = compute_retrieval and self.labels is not None
        self.save_retrieval_pairs = save_retrieval_pairs
        self.include_label_strings = include_label_strings

        if self.compute_retrieval:
            if self.labels is None or self.y is None:
                raise ValueError("compute_retrieval requires labels and y to be loaded")
            # Database = mean embedding across all trials per unique word
            # (canonical convention; see utils.retrieval).  Previously this used
            # the first occurrence of each word; the mean is more stable and
            # matches the shared retrieval procedure used elsewhere.
            unique_labels, self._retrieval_db_embeds_raw = mean_embedding_per_word(
                self.y, self.labels
            )
            self._retrieval_db_word_idx = np.array(
                [self.word_to_index[label] for label in unique_labels],
                dtype=self._index_dtype,
            )
        else:
            self._retrieval_db_embeds_raw = None
            self._retrieval_db_word_idx = None

        self.all_regressor_weights=[]
        self.all_regressor_intercept=[]
        self.all_test_score=[]
        self.all_train_score=[]
        self.all_chance=[]
        self.all_top_k_accuracy={k: [] for k in top_k_values}
        self.all_retrieval_top1=[]
        self.all_retrieval_top3=[]
        self.all_retrieval_top5=[]
        self.all_retrieval_chance_top1=[]
        self.all_retrieval_chance_top3=[]
        self.all_retrieval_chance_top5=[]
        self.all_retrieval_category_top1=[]
        self.all_retrieval_category_chance_top1=[]
        self.all_retrieval_pairs=[]
        self.all_retrieval_word_balanced_acc=[]
        self.all_retrieval_chance_word_balanced_acc=[]
        self.all_retrieval_word_f1=[]
        self.all_retrieval_chance_word_f1=[]
        self.all_retrieval_category_balanced_acc=[]
        self.all_retrieval_category_chance_balanced_acc=[]
        self.all_retrieval_category_f1=[]
        self.all_retrieval_category_chance_f1=[]

        if parallel is None:
            # Sequential processing
            results = [self._fit(i) for i in range(self.n_epochs)]
        else:
            # Parallel processing. In Windows/Jupyter sessions, pickling the live
            # notebook-loaded class can fail, so fall back to sequential execution.
            try:
                with Pool(processes=parallel, initializer=_worker_init) as pool:
                    results=pool.map(self._fit,range(self.n_epochs))
            except Exception as exc:
                if 'pickle' not in str(exc).lower():
                    raise
                warnings.warn(
                    'Parallel regression failed due to a notebook multiprocessing '
                    'pickling issue; falling back to sequential execution.',
                    RuntimeWarning,
                )
                results = [self._fit(i) for i in range(self.n_epochs)]
        
        all_regressor_weights=[]
        all_regressor_intercept=[]
        all_test_score=[]
        all_train_score=[]
        all_cosine_sim=[]
        all_train_cosine_sim=[]
        all_chance=[]
        all_top_k_accuracy={k: [] for k in top_k_values}
        all_retrieval_top1=[]
        all_retrieval_top3=[]
        all_retrieval_top5=[]
        all_retrieval_chance_top1=[]
        all_retrieval_chance_top3=[]
        all_retrieval_chance_top5=[]
        all_retrieval_category_top1=[]
        all_retrieval_category_chance_top1=[]
        all_retrieval_pairs=[]
        all_retrieval_word_balanced_acc=[]
        all_retrieval_chance_word_balanced_acc=[]
        all_retrieval_word_f1=[]
        all_retrieval_chance_word_f1=[]
        all_retrieval_category_balanced_acc=[]
        all_retrieval_category_chance_balanced_acc=[]
        all_retrieval_category_f1=[]
        all_retrieval_category_chance_f1=[]
        all_retrieval_category_indep_top1=[]
        all_retrieval_category_indep_chance_top1=[]
        all_retrieval_category_indep_balanced_acc=[]
        all_retrieval_category_indep_chance_balanced_acc=[]
        all_retrieval_category_indep_f1=[]
        all_retrieval_chance_category_indep_f1=[]

        for result in results:
            all_regressor_weights.append(result[0])
            all_regressor_intercept.append(result[1])
            all_test_score.append(result[2])
            all_train_score.append(result[3])
            all_chance.append(result[4])
            all_cosine_sim.append(result[19])
            all_train_cosine_sim.append(result[20])
            if compute_top_k_accuracy:
                for k_idx, k in enumerate(top_k_values):
                    all_top_k_accuracy[k].append(result[5][k_idx])
            if self.compute_retrieval:
                all_retrieval_top1.append(result[6])
                all_retrieval_chance_top1.append(result[7])
                all_retrieval_category_top1.append(result[8])
                all_retrieval_category_chance_top1.append(result[9])
                if self.save_retrieval_pairs:
                    all_retrieval_pairs.extend(result[10])
                all_retrieval_word_balanced_acc.append(result[11])
                all_retrieval_chance_word_balanced_acc.append(result[12])
                all_retrieval_word_f1.append(result[13])
                all_retrieval_chance_word_f1.append(result[14])
                all_retrieval_category_balanced_acc.append(result[15])
                all_retrieval_category_chance_balanced_acc.append(result[16])
                all_retrieval_category_f1.append(result[17])
                all_retrieval_category_chance_f1.append(result[18])
                all_retrieval_top3.append(result[21])
                all_retrieval_chance_top3.append(result[22])
                all_retrieval_top5.append(result[23])
                all_retrieval_chance_top5.append(result[24])
                all_retrieval_category_indep_top1.append(result[25])
                all_retrieval_category_indep_chance_top1.append(result[26])
                all_retrieval_category_indep_balanced_acc.append(result[27])
                all_retrieval_category_indep_chance_balanced_acc.append(result[28])
                all_retrieval_category_indep_f1.append(result[29])
                all_retrieval_chance_category_indep_f1.append(result[30])

        self.all_regressor_weights=all_regressor_weights
        self.all_regressor_intercept=np.array(all_regressor_intercept)
        self.all_test_score=np.array(all_test_score)
        self.all_train_score=np.array(all_train_score)
        self.all_cosine_sim=np.array(all_cosine_sim)
        self.all_train_cosine_sim=np.array(all_train_cosine_sim)
        self.all_chance=np.array(all_chance)
        if compute_top_k_accuracy:
            for k in top_k_values:
                self.all_top_k_accuracy[k]=np.array(all_top_k_accuracy[k])
        if self.compute_retrieval:
            self.all_retrieval_top1=np.array(all_retrieval_top1)
            self.all_retrieval_top3=np.array(all_retrieval_top3)
            self.all_retrieval_top5=np.array(all_retrieval_top5)
            self.all_retrieval_chance_top1=np.array(all_retrieval_chance_top1)
            self.all_retrieval_chance_top3=np.array(all_retrieval_chance_top3)
            self.all_retrieval_chance_top5=np.array(all_retrieval_chance_top5)
            self.all_retrieval_category_top1=np.array(all_retrieval_category_top1)
            self.all_retrieval_category_chance_top1=np.array(all_retrieval_category_chance_top1)
            if self.save_retrieval_pairs:
                self.all_retrieval_pairs=all_retrieval_pairs
            self.all_retrieval_word_balanced_acc=np.array(all_retrieval_word_balanced_acc)
            self.all_retrieval_chance_word_balanced_acc=np.array(all_retrieval_chance_word_balanced_acc)
            self.all_retrieval_word_f1=np.array(all_retrieval_word_f1)
            self.all_retrieval_chance_word_f1=np.array(all_retrieval_chance_word_f1)
            self.all_retrieval_category_balanced_acc=np.array(all_retrieval_category_balanced_acc)
            self.all_retrieval_category_chance_balanced_acc=np.array(all_retrieval_category_chance_balanced_acc)
            self.all_retrieval_category_f1=np.array(all_retrieval_category_f1)
            self.all_retrieval_category_chance_f1=np.array(all_retrieval_category_chance_f1)
            self.all_retrieval_category_indep_top1=np.array(all_retrieval_category_indep_top1)
            self.all_retrieval_category_indep_chance_top1=np.array(all_retrieval_category_indep_chance_top1)
            self.all_retrieval_category_indep_balanced_acc=np.array(all_retrieval_category_indep_balanced_acc)
            self.all_retrieval_category_indep_chance_balanced_acc=np.array(all_retrieval_category_indep_chance_balanced_acc)
            self.all_retrieval_category_indep_f1=np.array(all_retrieval_category_indep_f1)
            self.all_retrieval_chance_category_indep_f1=np.array(all_retrieval_chance_category_indep_f1)

    def _fit(self, _):
        from sklearn.model_selection import KFold
        
        if self.X_to_use:
            all_regressor_weights=[]
            all_regressor_intercept=[]
            all_test_score=[]
            all_train_score=[]
            all_cosine_sim=[]
            all_train_cosine_sim=[]
            all_chance=[]
            all_top_k_accuracy=[[] for _ in self.top_k_values]
            all_retrieval_top1=[]
            all_retrieval_top3=[]
            all_retrieval_top5=[]
            all_retrieval_chance_top3=[]
            all_retrieval_chance_top5=[]
            all_retrieval_chance_top1=[]
            all_retrieval_category_top1=[]
            all_retrieval_category_chance_top1=[]
            all_retrieval_pairs=[]
            all_retrieval_word_balanced_acc=[]
            all_retrieval_chance_word_balanced_acc=[]
            all_retrieval_word_f1=[]
            all_retrieval_chance_word_f1=[]
            all_retrieval_category_balanced_acc=[]
            all_retrieval_category_chance_balanced_acc=[]
            all_retrieval_category_f1=[]
            all_retrieval_chance_category_f1=[]
            all_retrieval_category_indep_top1=[]
            all_retrieval_category_indep_chance_top1=[]
            all_retrieval_category_indep_balanced_acc=[]
            all_retrieval_category_indep_chance_balanced_acc=[]
            all_retrieval_category_indep_f1=[]
            all_retrieval_chance_category_indep_f1=[]

            for n_bin in range(len(self.X_to_use)):
                X = self.X_to_use[n_bin]
                y = self.y
                labels_test = None
                test_idx = np.array([], dtype=np.int32)
                
                if self.use_kfold:
                    # Use KFold
                    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=None)
                    fold_train_scores = []
                    fold_test_scores = []
                    fold_cosine_sims = []
                    fold_train_cosine_sims = []
                    fold_chances = []
                    fold_weights = []
                    fold_intercepts = []
                    fold_top_k_accs = [[] for _ in self.top_k_values]
                    fold_retrieval_top1 = []
                    fold_retrieval_top3 = []
                    fold_retrieval_top5 = []
                    fold_retrieval_chance_top3 = []
                    fold_retrieval_chance_top5 = []
                    fold_retrieval_chance_top1 = []
                    fold_retrieval_category_top1 = []
                    fold_retrieval_category_chance_top1 = []
                    fold_retrieval_word_balanced_acc = []
                    fold_retrieval_chance_word_balanced_acc = []
                    fold_retrieval_word_f1 = []
                    fold_retrieval_chance_word_f1 = []
                    fold_retrieval_category_balanced_acc = []
                    fold_retrieval_category_chance_balanced_acc = []
                    fold_retrieval_category_f1 = []
                    fold_retrieval_chance_category_f1 = []
                    fold_retrieval_category_indep_top1 = []
                    fold_retrieval_category_indep_chance_top1 = []
                    fold_retrieval_category_indep_balanced_acc = []
                    fold_retrieval_category_indep_chance_balanced_acc = []
                    fold_retrieval_category_indep_f1 = []
                    fold_retrieval_chance_category_indep_f1 = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        labels_test = self.labels[test_idx] if self.compute_retrieval else None
                        
                        # reduce the dimensionality for x and y separately
                        if self.x_reducer is not None:
                            X_train = self.x_reducer.fit_transform(X_train)
                            X_test = self.x_reducer.transform(X_test)

                        if self.y_reducer is not None:
                            y_train=self.y_reducer.fit_transform(y_train)
                            y_test=self.y_reducer.transform(y_test)

                        # regression
                        self.regressor.fit(X_train, y_train)
                        y_test_predict = self.regressor.predict(X_test)
                        y_train_predict = self.regressor.predict(X_train)
                        train_score=self.regressor.score(X_train, y_train)
                        test_score=self.regressor.score(X_test, y_test)

                        # Cosine similarity (direction-only goodness of fit)
                        fold_test_cosine = self._mean_cosine_similarity(y_test_predict, y_test)
                        fold_train_cosine = self._mean_cosine_similarity(y_train_predict, y_train)

                        # Compute top-k accuracy
                        if self.compute_top_k_accuracy:
                            top_k_accs = self._compute_top_k_accuracy(y_test_predict, y_test)
                            for k_idx in range(len(self.top_k_values)):
                                fold_top_k_accs[k_idx].append(top_k_accs[k_idx])

                        # Compute retrieval accuracy
                        if self.compute_retrieval:
                            retrieval = self._compute_retrieval_accuracy(y_test_predict, labels_test)
                            fold_retrieval_top1.append(retrieval["word_top1_acc"])
                            fold_retrieval_top3.append(retrieval["word_top3_acc"])
                            fold_retrieval_top5.append(retrieval["word_top5_acc"])
                            fold_retrieval_word_balanced_acc.append(retrieval["word_balanced_acc"])
                            fold_retrieval_word_f1.append(retrieval["word_f1"])
                            if retrieval["category_top1_acc"] is not None:
                                fold_retrieval_category_top1.append(float(retrieval["category_top1_acc"]))
                                fold_retrieval_category_balanced_acc.append(float(retrieval["category_balanced_acc"]))
                                fold_retrieval_category_f1.append(float(retrieval["category_f1"]))
                            else:
                                fold_retrieval_category_top1.append(np.nan)
                                fold_retrieval_category_balanced_acc.append(np.nan)
                                fold_retrieval_category_f1.append(np.nan)
                            if retrieval["category_indep_top1_acc"] is not None:
                                fold_retrieval_category_indep_top1.append(float(retrieval["category_indep_top1_acc"]))
                                fold_retrieval_category_indep_balanced_acc.append(float(retrieval["category_indep_balanced_acc"]))
                                fold_retrieval_category_indep_f1.append(float(retrieval["category_indep_f1"]))
                            else:
                                fold_retrieval_category_indep_top1.append(np.nan)
                                fold_retrieval_category_indep_balanced_acc.append(np.nan)
                                fold_retrieval_category_indep_f1.append(np.nan)

                            if self.save_retrieval_pairs:
                                pair_payload = {
                                    "bin_index": int(n_bin),
                                    "fold_index": int(fold_idx),
                                    "test_indices": test_idx.astype(np.int32, copy=False),
                                    "true_word_idx": retrieval["true_word_idx"],
                                    "pred_word_idx": retrieval["pred_word_idx"],
                                }
                                if retrieval.get("pred_cat_idx_indep") is not None:
                                    pair_payload["pred_category_idx_indep"] = retrieval["pred_cat_idx_indep"]
                                if self.include_label_strings:
                                    pair_payload["true_word_labels"] = retrieval["true_word_labels"]
                                    pair_payload["pred_word_labels"] = retrieval["pred_word_labels"]
                                all_retrieval_pairs.append(pair_payload)

                        # regression to shuffled neural activity
                        X_train_shuffle=np.random.permutation(X_train.flatten()).reshape(X_train.shape)
                        self.regressor.fit(X_train_shuffle, y_train)
                        if self.compute_retrieval:
                            y_shuffle_predict = self.regressor.predict(X_test)
                            chance_retrieval = self._compute_retrieval_accuracy(y_shuffle_predict, labels_test)
                            fold_retrieval_chance_top1.append(chance_retrieval["word_top1_acc"])
                            fold_retrieval_chance_top3.append(chance_retrieval["word_top3_acc"])
                            fold_retrieval_chance_top5.append(chance_retrieval["word_top5_acc"])
                            fold_retrieval_chance_word_balanced_acc.append(chance_retrieval["word_balanced_acc"])
                            fold_retrieval_chance_word_f1.append(chance_retrieval["word_f1"])
                            if chance_retrieval["category_top1_acc"] is not None:
                                fold_retrieval_category_chance_top1.append(float(chance_retrieval["category_top1_acc"]))
                                fold_retrieval_chance_category_balanced_acc.append(float(chance_retrieval["category_balanced_acc"]))
                                fold_retrieval_chance_category_f1.append(float(chance_retrieval["category_f1"]))
                            else:
                                fold_retrieval_category_chance_top1.append(np.nan)
                                fold_retrieval_chance_category_balanced_acc.append(np.nan)
                                fold_retrieval_chance_category_f1.append(np.nan)
                            if chance_retrieval["category_indep_top1_acc"] is not None:
                                fold_retrieval_category_indep_chance_top1.append(float(chance_retrieval["category_indep_top1_acc"]))
                                fold_retrieval_category_indep_chance_balanced_acc.append(float(chance_retrieval["category_indep_balanced_acc"]))
                                fold_retrieval_chance_category_indep_f1.append(float(chance_retrieval["category_indep_f1"]))
                            else:
                                fold_retrieval_category_indep_chance_top1.append(np.nan)
                                fold_retrieval_category_indep_chance_balanced_acc.append(np.nan)
                                fold_retrieval_chance_category_indep_f1.append(np.nan)
                        shuffle_score=self.regressor.score(X_test, y_test)
                        
                        fold_train_scores.append(train_score)
                        fold_test_scores.append(test_score)
                        fold_cosine_sims.append(fold_test_cosine)
                        fold_train_cosine_sims.append(fold_train_cosine)
                        fold_chances.append(shuffle_score)
                        if hasattr(self.regressor, 'coef_'):
                            fold_weights.append(self.regressor.coef_)
                        if hasattr(self.regressor, 'intercept_'):
                            fold_intercepts.append(self.regressor.intercept_)
                    
                    # Average across folds
                    all_train_score.append(np.mean(fold_train_scores))
                    all_test_score.append(np.mean(fold_test_scores))
                    all_cosine_sim.append(np.mean(fold_cosine_sims))
                    all_train_cosine_sim.append(np.mean(fold_train_cosine_sims))
                    all_chance.append(np.mean(fold_chances))
                    if fold_weights:
                        all_regressor_weights.append(np.mean(fold_weights, axis=0))
                    if fold_intercepts:
                        all_regressor_intercept.append(np.mean(fold_intercepts, axis=0))
                    
                    if self.compute_top_k_accuracy:
                        for k_idx in range(len(self.top_k_values)):
                            all_top_k_accuracy[k_idx].append(np.mean(fold_top_k_accs[k_idx]))
                    if self.compute_retrieval:
                        all_retrieval_top1.append(np.mean(fold_retrieval_top1))
                        all_retrieval_top3.append(np.mean(fold_retrieval_top3))
                        all_retrieval_top5.append(np.mean(fold_retrieval_top5))
                        all_retrieval_chance_top3.append(np.mean(fold_retrieval_chance_top3))
                        all_retrieval_chance_top5.append(np.mean(fold_retrieval_chance_top5))
                        all_retrieval_chance_top1.append(np.mean(fold_retrieval_chance_top1))
                        all_retrieval_word_balanced_acc.append(np.mean(fold_retrieval_word_balanced_acc))
                        all_retrieval_chance_word_balanced_acc.append(np.mean(fold_retrieval_chance_word_balanced_acc))
                        all_retrieval_word_f1.append(np.mean(fold_retrieval_word_f1))
                        all_retrieval_chance_word_f1.append(np.mean(fold_retrieval_chance_word_f1))
                        if len(fold_retrieval_category_top1) > 0:
                            all_retrieval_category_top1.append(np.mean(fold_retrieval_category_top1))
                            all_retrieval_category_chance_top1.append(np.mean(fold_retrieval_category_chance_top1))
                            all_retrieval_category_balanced_acc.append(np.mean(fold_retrieval_category_balanced_acc))
                            all_retrieval_category_chance_balanced_acc.append(np.mean(fold_retrieval_chance_category_balanced_acc))
                            all_retrieval_category_f1.append(np.mean(fold_retrieval_category_f1))
                            all_retrieval_chance_category_f1.append(np.mean(fold_retrieval_chance_category_f1))
                        else:
                            all_retrieval_category_top1.append(np.nan)
                            all_retrieval_category_chance_top1.append(np.nan)
                            all_retrieval_category_balanced_acc.append(np.nan)
                            all_retrieval_category_chance_balanced_acc.append(np.nan)
                            all_retrieval_category_f1.append(np.nan)
                            all_retrieval_chance_category_f1.append(np.nan)
                        if len(fold_retrieval_category_indep_top1) > 0:
                            all_retrieval_category_indep_top1.append(np.mean(fold_retrieval_category_indep_top1))
                            all_retrieval_category_indep_chance_top1.append(np.mean(fold_retrieval_category_indep_chance_top1))
                            all_retrieval_category_indep_balanced_acc.append(np.mean(fold_retrieval_category_indep_balanced_acc))
                            all_retrieval_category_indep_chance_balanced_acc.append(np.mean(fold_retrieval_category_indep_chance_balanced_acc))
                            all_retrieval_category_indep_f1.append(np.mean(fold_retrieval_category_indep_f1))
                            all_retrieval_chance_category_indep_f1.append(np.mean(fold_retrieval_chance_category_indep_f1))
                        else:
                            all_retrieval_category_indep_top1.append(np.nan)
                            all_retrieval_category_indep_chance_top1.append(np.nan)
                            all_retrieval_category_indep_balanced_acc.append(np.nan)
                            all_retrieval_category_indep_chance_balanced_acc.append(np.nan)
                            all_retrieval_category_indep_f1.append(np.nan)
                            all_retrieval_chance_category_indep_f1.append(np.nan)
                    
                else:
                    # Use random train_test_split (original behavior)
                    if self.compute_retrieval:
                        indices = np.arange(len(X))
                        train_idx, test_idx = train_test_split(indices, test_size=self.split)
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        labels_test = self.labels[test_idx]
                    else:
                        X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=self.split)
                    
                    # reduce the dimensionality for x and y separately
                    if self.x_reducer is not None:
                        X_train = self.x_reducer.fit_transform(X_train)
                        X_test = self.x_reducer.transform(X_test)

                    if self.y_reducer is not None:
                        y_train=self.y_reducer.fit_transform(y_train)
                        y_test=self.y_reducer.transform(y_test)

                    # regression
                    self.regressor.fit(X_train, y_train)
                    y_test_predict = self.regressor.predict(X_test)
                    y_train_predict = self.regressor.predict(X_train)
                    train_score=self.regressor.score(X_train, y_train)
                    test_score=self.regressor.score(X_test, y_test)

                    # Cosine similarity (direction-only goodness of fit)
                    test_cosine = self._mean_cosine_similarity(y_test_predict, y_test)
                    train_cosine = self._mean_cosine_similarity(y_train_predict, y_train)

                    # Compute top-k accuracy
                    if self.compute_top_k_accuracy:
                        top_k_accs = self._compute_top_k_accuracy(y_test_predict, y_test)
                        for k_idx in range(len(self.top_k_values)):
                            all_top_k_accuracy[k_idx].append(top_k_accs[k_idx])

                    # Compute retrieval accuracy
                    if self.compute_retrieval:
                        retrieval = self._compute_retrieval_accuracy(y_test_predict, labels_test)
                        all_retrieval_top1.append(retrieval["word_top1_acc"])
                        all_retrieval_top3.append(retrieval["word_top3_acc"])
                        all_retrieval_top5.append(retrieval["word_top5_acc"])
                        all_retrieval_word_balanced_acc.append(retrieval["word_balanced_acc"])
                        all_retrieval_word_f1.append(retrieval["word_f1"])
                        if retrieval["category_top1_acc"] is not None:
                            all_retrieval_category_top1.append(float(retrieval["category_top1_acc"]))
                            all_retrieval_category_balanced_acc.append(float(retrieval["category_balanced_acc"]))
                            all_retrieval_category_f1.append(float(retrieval["category_f1"]))
                        else:
                            all_retrieval_category_top1.append(np.nan)
                            all_retrieval_category_balanced_acc.append(np.nan)
                            all_retrieval_category_f1.append(np.nan)
                        if retrieval["category_indep_top1_acc"] is not None:
                            all_retrieval_category_indep_top1.append(float(retrieval["category_indep_top1_acc"]))
                            all_retrieval_category_indep_balanced_acc.append(float(retrieval["category_indep_balanced_acc"]))
                            all_retrieval_category_indep_f1.append(float(retrieval["category_indep_f1"]))
                        else:
                            all_retrieval_category_indep_top1.append(np.nan)
                            all_retrieval_category_indep_balanced_acc.append(np.nan)
                            all_retrieval_category_indep_f1.append(np.nan)

                        if self.save_retrieval_pairs:
                            pair_payload = {
                                "bin_index": int(n_bin),
                                "fold_index": -1,
                                "test_indices": test_idx.astype(np.int32, copy=False),
                                "true_word_idx": retrieval["true_word_idx"],
                                "pred_word_idx": retrieval["pred_word_idx"],
                            }
                            if retrieval.get("pred_cat_idx_indep") is not None:
                                pair_payload["pred_category_idx_indep"] = retrieval["pred_cat_idx_indep"]
                            if self.include_label_strings:
                                pair_payload["true_word_labels"] = retrieval["true_word_labels"]
                                pair_payload["pred_word_labels"] = retrieval["pred_word_labels"]
                            all_retrieval_pairs.append(pair_payload)

                    # regression to shuffled neural activity
                    X_train_shuffle=np.random.permutation(X_train.flatten()).reshape(X_train.shape)
                    self.regressor.fit(X_train_shuffle, y_train)
                    if self.compute_retrieval:
                        y_shuffle_predict = self.regressor.predict(X_test)
                        chance_retrieval = self._compute_retrieval_accuracy(y_shuffle_predict, labels_test)
                        all_retrieval_chance_top1.append(chance_retrieval["word_top1_acc"])
                        all_retrieval_chance_top3.append(chance_retrieval["word_top3_acc"])
                        all_retrieval_chance_top5.append(chance_retrieval["word_top5_acc"])
                        all_retrieval_chance_word_balanced_acc.append(chance_retrieval["word_balanced_acc"])
                        all_retrieval_chance_word_f1.append(chance_retrieval["word_f1"])
                        if chance_retrieval["category_top1_acc"] is not None:
                            all_retrieval_category_chance_top1.append(float(chance_retrieval["category_top1_acc"]))
                            all_retrieval_category_chance_balanced_acc.append(float(chance_retrieval["category_balanced_acc"]))
                            all_retrieval_chance_category_f1.append(float(chance_retrieval["category_f1"]))
                        else:
                            all_retrieval_category_chance_top1.append(np.nan)
                            all_retrieval_category_chance_balanced_acc.append(np.nan)
                            all_retrieval_chance_category_f1.append(np.nan)
                        if chance_retrieval["category_indep_top1_acc"] is not None:
                            all_retrieval_category_indep_chance_top1.append(float(chance_retrieval["category_indep_top1_acc"]))
                            all_retrieval_category_indep_chance_balanced_acc.append(float(chance_retrieval["category_indep_balanced_acc"]))
                            all_retrieval_chance_category_indep_f1.append(float(chance_retrieval["category_indep_f1"]))
                        else:
                            all_retrieval_category_indep_chance_top1.append(np.nan)
                            all_retrieval_category_indep_chance_balanced_acc.append(np.nan)
                            all_retrieval_chance_category_indep_f1.append(np.nan)
                    shuffle_score=self.regressor.score(X_test, y_test)

                    if hasattr(self.regressor, 'coef_'):
                        all_regressor_weights.append(self.regressor.coef_)
                    if hasattr(self.regressor, 'intercept_'):
                        all_regressor_intercept.append(self.regressor.intercept_)
                    all_test_score.append(test_score)
                    all_train_score.append(train_score)
                    all_chance.append(shuffle_score)
                    all_cosine_sim.append(test_cosine)
                    all_train_cosine_sim.append(train_cosine)

            return (all_regressor_weights,           # 0
                    all_regressor_intercept,          # 1
                    all_test_score,                   # 2
                    all_train_score,                  # 3
                    all_chance,                       # 4
                    all_top_k_accuracy,               # 5
                    all_retrieval_top1,               # 6
                    all_retrieval_chance_top1,        # 7
                    all_retrieval_category_top1,      # 8
                    all_retrieval_category_chance_top1, # 9
                    all_retrieval_pairs,              # 10
                    all_retrieval_word_balanced_acc,  # 11
                    all_retrieval_chance_word_balanced_acc, # 12
                    all_retrieval_word_f1,            # 13
                    all_retrieval_chance_word_f1,     # 14
                    all_retrieval_category_balanced_acc, # 15
                    all_retrieval_category_chance_balanced_acc, # 16
                    all_retrieval_category_f1,        # 17
                    all_retrieval_chance_category_f1, # 18
                    all_cosine_sim,                   # 19
                    all_train_cosine_sim,             # 20
                    all_retrieval_top3,               # 21
                    all_retrieval_chance_top3,        # 22
                    all_retrieval_top5,               # 23
                    all_retrieval_chance_top5,        # 24
                    all_retrieval_category_indep_top1,              # 25
                    all_retrieval_category_indep_chance_top1,       # 26
                    all_retrieval_category_indep_balanced_acc,      # 27
                    all_retrieval_category_indep_chance_balanced_acc, # 28
                    all_retrieval_category_indep_f1,                # 29
                    all_retrieval_chance_category_indep_f1)         # 30

    def predict(self, X):
        if self.x_reducer is not None:
            X = self.x_reducer.transform(X)
        return self.regressor.predict(X)

    def score(self, X, y):
        if self.x_reducer is not None:
            X = self.x_reducer.transform(X)
        if self.y_reducer is not None:
            y = self.y_reducer.transform(y)
        return self.regressor.score(X, y)

    @staticmethod
    def _mean_cosine_similarity(y_pred, y_true):
        """
        Mean cosine similarity between predicted and true embeddings,
        computed on mean-centered residuals.

        Mean-centering (subtracting the column-wise mean of y_true) removes
        the dominant shared direction that is common to all embeddings.
        Without this, early ViT layers — where all word vectors cluster
        tightly (pairwise cosine ~0.99) — yield artificially inflated cosine
        scores (~0.99) even for a model that predicts the centroid for every
        sample.  Mean-centering ensures the metric reflects word-discriminating
        signal, matching the same convention used in _compute_retrieval_accuracy.

        Range after centering: -1 (opposite residuals) to +1 (identical
        residuals); 0 = orthogonal / no discriminating alignment.
        """
        mean = y_true.mean(axis=0)
        y_pred_c = y_pred - mean
        y_true_c = y_true - mean
        pred_norm = np.linalg.norm(y_pred_c, axis=1, keepdims=True) + 1e-10
        true_norm = np.linalg.norm(y_true_c, axis=1, keepdims=True) + 1e-10
        cos = np.sum((y_pred_c / pred_norm) * (y_true_c / true_norm), axis=1)
        return float(np.mean(cos))

    def _get_closest_predictions(self, y_pred):
        # Find the closest predictions in the training set
        closest = []
        
        if self.y_reducer is not None:
            y_pred = self.y_reducer.inverse_transform(y_pred)

        for pred in y_pred:

            if self._closest=='l2':
                idx = np.sum((self.y - pred) ** 2, axis=1).argmin()
            elif self._closest=='l1':
                idx = np.sum(np.abs(self.y - pred), axis=1).argmin()
            else:
                idx=None
            closest.append(self.y[idx])
        return np.array(closest)
    
    def _compute_top_k_accuracy(self, y_pred, y_test):
        """
        Compute top-k accuracy for regression predictions (vectorized version).
        
        For each prediction, find the k nearest embeddings in the full vocabulary (self.y)
        and check if the true target embedding is among them.
        
        Parameters:
        -----------
        y_pred : array-like, shape (n_samples, embedding_dim)
            Predicted embeddings from the regressor
        y_test : array-like, shape (n_samples, embedding_dim)
            True target embeddings for test samples
            
        Returns:
        --------
        top_k_accs : list of float
            Top-k accuracies for each k value in self.top_k_values
        """
        # Inverse transform predictions if y_reducer was used
        if self.y_reducer is not None:
            y_pred = self.y_reducer.inverse_transform(y_pred)
            y_test = self.y_reducer.inverse_transform(y_test)
        
        # Get unique embeddings from vocabulary to avoid redundant distance calculations
        # This is much more efficient when self.y contains many repeated embeddings
        # Use pandas to get unique rows since np.unique doesn't support axis in older versions
        unique_embeddings = pd.DataFrame(self.y).drop_duplicates().values
        
        # Vectorized distance computation: distance from each prediction to UNIQUE vocabulary embeddings
        # Shape: (n_predictions, n_unique_embeddings)
        if self._closest == 'l2':
            # Broadcasting: (n_pred, 1, dim) - (1, n_unique, dim) = (n_pred, n_unique, dim)
            distances_to_vocab = np.sum((y_pred[:, np.newaxis, :] - unique_embeddings[np.newaxis, :, :]) ** 2, axis=2)
        elif self._closest == 'l1':
            distances_to_vocab = np.sum(np.abs(y_pred[:, np.newaxis, :] - unique_embeddings[np.newaxis, :, :]), axis=2)
        elif self._closest == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-10)
            y_vocab_norm = unique_embeddings / (np.linalg.norm(unique_embeddings, axis=1, keepdims=True) + 1e-10)
            similarities = y_pred_norm @ y_vocab_norm.T  # (n_pred, n_unique)
            distances_to_vocab = 1 - similarities
        else:
            # Default to L2
            distances_to_vocab = np.sum((y_pred[:, np.newaxis, :] - unique_embeddings[np.newaxis, :, :]) ** 2, axis=2)
        
        # Distance from each prediction to its true target
        # Shape: (n_predictions,)
        if self._closest == 'l2':
            distances_to_true = np.sum((y_pred - y_test) ** 2, axis=1)
        elif self._closest == 'l1':
            distances_to_true = np.sum(np.abs(y_pred - y_test), axis=1)
        elif self._closest == 'cosine':
            y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-10)
            y_test_norm = y_test / (np.linalg.norm(y_test, axis=1, keepdims=True) + 1e-10)
            similarities = np.sum(y_pred_norm * y_test_norm, axis=1)
            distances_to_true = 1 - similarities
        else:
            distances_to_true = np.sum((y_pred - y_test) ** 2, axis=1)
        
        # For each prediction, count how many unique vocabulary embeddings are closer than the true target
        # Shape: (n_predictions, n_unique_embeddings)
        is_closer = distances_to_vocab <= distances_to_true[:, np.newaxis]
        ranks = is_closer.sum(axis=1)   # +1 because rank starts at 1
        
        # Compute top-k accuracy for all k values
        top_k_accs = []
        for k in self.top_k_values:
            # True target is in top-k if its rank <= k
            in_top_k = (ranks <= k).sum()
            top_k_accs.append(in_top_k / len(y_pred) if len(y_pred) > 0 else 0.0)
        
        return top_k_accs

    def _compute_retrieval_accuracy(self, y_pred, test_labels):
        """
        Retrieval accuracy: predicted embeddings (queries) vs a vocabulary
        database of one representative embedding per unique word.

        y_pred is in reduced (PCA) space when y_reducer is set; the database
        is projected into that same space to avoid inverse_transform overhead.

        Returns top1_accuracy.
        """
        if self._retrieval_db_embeds_raw is None or self._retrieval_db_word_idx is None:
            raise RuntimeError("Retrieval database not initialized. Call fit with compute_retrieval=True.")

        db_embeds = self._retrieval_db_embeds_raw
        db_word_idx = self._retrieval_db_word_idx

        # Project database into the same space as y_pred.
        # This avoids calling inverse_transform (50D → 300D+) on every query.
        if self.y_reducer is not None:
            db_embeds = self.y_reducer.transform(db_embeds)   # original → reduced

        # Memory-efficient L2: ||a-b||² = ||a||² + ||b||² - 2 a·b
        # Mean-center: subtract the database centroid from both the database and the
        # predictions so that retrieval operates on deviations from the mean, preventing
        # the model from "cheating" by always predicting the centroid.
        db_embeds, y_pred, db_mean = mean_center_db(db_embeds, y_pred)

        # Avoids the (n_test, n_db, dim) intermediate tensor entirely.
        if self._closest == 'cosine':
            pred_norm = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-10)
            db_norm = db_embeds / (np.linalg.norm(db_embeds, axis=1, keepdims=True) + 1e-10)
            distances = 1 - pred_norm @ db_norm.T
        else:
            sq_pred = np.sum(y_pred ** 2, axis=1, keepdims=True)   # (n_test, 1)
            sq_db   = np.sum(db_embeds ** 2, axis=1)               # (n_db,)
            distances = sq_pred + sq_db - 2 * (y_pred @ db_embeds.T)  # (n_test, n_db)

        sorted_idx = np.argsort(distances, axis=1)

        nearest_db = sorted_idx[:, 0]
        pred_word_idx = db_word_idx[nearest_db].astype(self._index_dtype, copy=False)
        true_word_idx = np.array(
            [self.word_to_index[label] for label in test_labels],
            dtype=self._index_dtype,
        )

        top1_acc = np.mean(pred_word_idx == true_word_idx)

        # Top-3 / Top-5: true word appears anywhere in the K nearest DB entries
        top3_word_idx = db_word_idx[sorted_idx[:, :3]]   # (n_test, 3)
        top5_word_idx = db_word_idx[sorted_idx[:, :5]]   # (n_test, 5)
        top3_acc = float(np.mean(
            np.any(top3_word_idx == true_word_idx[:, None], axis=1)
        ))
        top5_acc = float(np.mean(
            np.any(top5_word_idx == true_word_idx[:, None], axis=1)
        ))
        _word_labels = np.unique(true_word_idx).tolist()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            word_balanced_acc = float(balanced_accuracy_score(true_word_idx, pred_word_idx))
        word_f1 = float(f1_score(true_word_idx, pred_word_idx, average='macro',
                                 labels=_word_labels, zero_division=0))

        category_top1_acc = None
        category_balanced_acc = None
        category_f1 = None
        if self.word_index_to_category_index is not None:
            pred_cat = self.word_index_to_category_index[pred_word_idx]
            true_cat = self.word_index_to_category_index[true_word_idx]
            _cat_labels = np.unique(true_cat).tolist()
            category_top1_acc = np.mean(pred_cat == true_cat)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                category_balanced_acc = float(balanced_accuracy_score(true_cat, pred_cat))
            category_f1 = float(f1_score(true_cat, pred_cat, average='macro',
                                         labels=_cat_labels, zero_division=0))

        # ── Independent category retrieval (centroid-level in embedding space) ──
        category_indep_top1_acc = None
        category_indep_balanced_acc = None
        category_indep_f1 = None
        pred_cat_idx_indep = None
        if self.word_index_to_category_index is not None:
            n_cats = len(self.index_to_category)
            true_cat = self.word_index_to_category_index[true_word_idx]
            # Build category centroids from word embeddings (already mean-centered)
            cat_centroids = np.zeros((n_cats, db_embeds.shape[1]), dtype=np.float64)
            cat_word_counts = np.zeros(n_cats, dtype=np.int64)
            for wi in range(len(db_word_idx)):
                ci = self.word_index_to_category_index[db_word_idx[wi]]
                cat_centroids[ci] += db_embeds[wi]
                cat_word_counts[ci] += 1
            valid_c = cat_word_counts > 0
            cat_centroids[valid_c] /= cat_word_counts[valid_c, np.newaxis]

            if self._closest == 'cosine':
                pred_n = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-10)
                cat_n = cat_centroids / (np.linalg.norm(cat_centroids, axis=1, keepdims=True) + 1e-10)
                cat_dists = 1 - pred_n @ cat_n.T
            else:
                sq_p = np.sum(y_pred ** 2, axis=1, keepdims=True)
                sq_c = np.sum(cat_centroids ** 2, axis=1)
                cat_dists = sq_p + sq_c - 2 * (y_pred @ cat_centroids.T)

            pred_cat_idx_indep = np.argmin(cat_dists, axis=1).astype(np.int32)
            _cat_labels = np.unique(true_cat).tolist()
            category_indep_top1_acc = float(np.mean(pred_cat_idx_indep == true_cat))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                category_indep_balanced_acc = float(balanced_accuracy_score(true_cat, pred_cat_idx_indep))
            category_indep_f1 = float(f1_score(true_cat, pred_cat_idx_indep, average='macro',
                                                labels=_cat_labels, zero_division=0))

        out = {
            "word_top1_acc": float(top1_acc),
            "word_top3_acc": top3_acc,
            "word_top5_acc": top5_acc,
            "word_balanced_acc": word_balanced_acc,
            "word_f1": word_f1,
            "category_top1_acc": (None if category_top1_acc is None else float(category_top1_acc)),
            "category_balanced_acc": category_balanced_acc,
            "category_f1": category_f1,
            "category_indep_top1_acc": (None if category_indep_top1_acc is None else float(category_indep_top1_acc)),
            "category_indep_balanced_acc": category_indep_balanced_acc,
            "category_indep_f1": category_indep_f1,
            "pred_word_idx": pred_word_idx,
            "true_word_idx": true_word_idx,
            "pred_cat_idx_indep": pred_cat_idx_indep,
        }

        if self.include_label_strings:
            out["pred_word_labels"] = self.index_to_word[pred_word_idx]
            out["true_word_labels"] = self.index_to_word[true_word_idx]

        return out
