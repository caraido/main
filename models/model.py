import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import  Dataset
from multiprocessing import Pool
from utils.utils import reformat
from sklearn.model_selection import train_test_split

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
            with Pool(processes=parallel) as pool:
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
        self.all_chance=[]
        self.all_top_k_accuracy={}

    def load_data(self, data, y, split=0.3, n_bins_history=10):
        # here data is the neural activity
        # y is the word/picture/any embeddings
        self.data = data
        self.y = y
        self.split = split
        self.n_bins_history = n_bins_history

        self.X_to_use=reformat(self.data,
                               self.n_bins_history)
        self.n_bins = len(self.X_to_use)

    
    def fit(self, n_epochs=50, parallel=None, closest='l2', use_kfold=False, n_splits=5, 
            compute_top_k_accuracy=True, top_k_values=[1, 3, 5, 10]):
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
        """
        self.n_epochs=n_epochs
        self._closest=closest
        self.use_kfold=use_kfold
        self.n_splits=n_splits
        self.compute_top_k_accuracy=compute_top_k_accuracy
        self.top_k_values=top_k_values

        self.all_regressor_weights=[]
        self.all_regressor_intercept=[]
        self.all_test_score=[]
        self.all_train_score=[]
        self.all_chance=[]
        self.all_top_k_accuracy={k: [] for k in top_k_values}

        if parallel is None:
            # Sequential processing
            results = [self._fit(i) for i in range(self.n_epochs)]
        else:
            # Parallel processing
            with Pool(processes=parallel) as pool:
                results=pool.map(self._fit,range(self.n_epochs))
        
        all_regressor_weights=[]
        all_regressor_intercept=[]
        all_test_score=[]
        all_train_score=[]
        all_chance=[]
        all_top_k_accuracy={k: [] for k in top_k_values}

        for result in results:
            all_regressor_weights.append(result[0])
            all_regressor_intercept.append(result[1])
            all_test_score.append(result[2])
            all_train_score.append(result[3])
            all_chance.append(result[4])
            if compute_top_k_accuracy:
                for k_idx, k in enumerate(top_k_values):
                    all_top_k_accuracy[k].append(result[5][k_idx])

        self.all_regressor_weights=all_regressor_weights
        self.all_regressor_intercept=np.array(all_regressor_intercept)
        self.all_test_score=np.array(all_test_score)
        self.all_train_score=np.array(all_train_score)
        self.all_chance=np.array(all_chance)
        if compute_top_k_accuracy:
            for k in top_k_values:
                self.all_top_k_accuracy[k]=np.array(all_top_k_accuracy[k])

    def _fit(self, _):
        from sklearn.model_selection import KFold
        
        if self.X_to_use:
            all_regressor_weights=[]
            all_regressor_intercept=[]
            all_test_score=[]
            all_train_score=[]
            all_chance=[]
            #all_ranked_prediction=[]
            all_ranked_accuracy=[]
            all_top_k_accuracy=[[] for _ in self.top_k_values]

            for n_bin in range(len(self.X_to_use)):
                X = self.X_to_use[n_bin]
                y = self.y
                
                if self.use_kfold:
                    # Use KFold
                    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=None)
                    fold_train_scores = []
                    fold_test_scores = []
                    fold_chances = []
                    fold_weights = []
                    fold_intercepts = []
                    fold_top_k_accs = [[] for _ in self.top_k_values]
                    
                    for train_idx, test_idx in kf.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
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
                        y_test_predict_closest = self._get_closest_predictions(y_test_predict)
                        train_score=self.regressor.score(X_train, y_train)
                        test_score=self.regressor.score(X_test, y_test)
                        
                        # Compute top-k accuracy
                        if self.compute_top_k_accuracy:
                            top_k_accs = self._compute_top_k_accuracy(y_test_predict, y_test)
                            for k_idx in range(len(self.top_k_values)):
                                fold_top_k_accs[k_idx].append(top_k_accs[k_idx])

                        # regression to shuffled neural activity
                        X_train_shuffle=np.random.permutation(X_train.flatten()).reshape(X_train.shape)
                        self.regressor.fit(X_train_shuffle, y_train)
                        shuffle_score=self.regressor.score(X_test, y_test)
                        
                        fold_train_scores.append(train_score)
                        fold_test_scores.append(test_score)
                        fold_chances.append(shuffle_score)
                        if hasattr(self.regressor, 'coef_'):
                            fold_weights.append(self.regressor.coef_)
                        if hasattr(self.regressor, 'intercept_'):
                            fold_intercepts.append(self.regressor.intercept_)
                    
                    # Average across folds
                    all_train_score.append(np.mean(fold_train_scores))
                    all_test_score.append(np.mean(fold_test_scores))
                    all_chance.append(np.mean(fold_chances))
                    if fold_weights:
                        all_regressor_weights.append(np.mean(fold_weights, axis=0))
                    if fold_intercepts:
                        all_regressor_intercept.append(np.mean(fold_intercepts, axis=0))
                    
                    if self.compute_top_k_accuracy:
                        for k_idx in range(len(self.top_k_values)):
                            all_top_k_accuracy[k_idx].append(np.mean(fold_top_k_accs[k_idx]))
                    
                else:
                    # Use random train_test_split (original behavior)
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
                    train_score=self.regressor.score(X_train, y_train)
                    test_score=self.regressor.score(X_test, y_test)
                    
                    # Compute top-k accuracy
                    if self.compute_top_k_accuracy:
                        top_k_accs = self._compute_top_k_accuracy(y_test_predict, y_test)
                        for k_idx in range(len(self.top_k_values)):
                            all_top_k_accuracy[k_idx].append(top_k_accs[k_idx])

                    # regression to shuffled neural activity
                    X_train_shuffle=np.random.permutation(X_train.flatten()).reshape(X_train.shape)
                    self.regressor.fit(X_train_shuffle, y_train)
                    shuffle_score=self.regressor.score(X_test, y_test)

                    if hasattr(self.regressor, 'coef_'):
                        all_regressor_weights.append(self.regressor.coef_)
                    if hasattr(self.regressor, 'intercept_'):
                        all_regressor_intercept.append(self.regressor.intercept_)
                    all_test_score.append(test_score)
                    all_train_score.append(train_score)
                    all_chance.append(shuffle_score)

            return (all_regressor_weights, 
                    all_regressor_intercept, 
                    all_test_score,
                    all_train_score, 
                    all_chance, 
                    all_top_k_accuracy)

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
