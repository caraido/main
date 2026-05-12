# -*- coding: utf-8 -*-
"""utils.decoder -- the GeneralDecoder class."""

import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import matplotlib.pyplot as plt

class GeneralDecoder:
    """
    Reusable decoder wrapper for time-bin-wise neural decoding experiments.

    Parameters
    ----------
    extractor : sklearn-like transformer
        Feature extractor with fit/transform methods.
    decoder : sklearn-like estimator
        Classifier with fit/predict methods.
    """

    def __init__(self, extractor, decoder):
        self.extractor = extractor
        self.decoder = decoder
        self.X_to_use = None  # list of [n_trials, n_features] matrices
        self.y = None
        self.scaler = StandardScaler()

        self.train_accuracy = None
        self.test_accuracy = None
        self.chance = None
        self.mean_train_accuracy = None
        self.mean_test_accuracy = None
        self.mean_chance = None
        self.std_train_accuracy = None
        self.std_test_accuracy = None
        self.std_chance = None

    def _evaluate_split(self, X, y, test_size):
        """Run one train/test split and return train, test, and shuffle-baseline accuracy."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.extractor.fit(X_train, y_train)
        X_train_low = self.extractor.transform(X_train)
        X_test_low = self.extractor.transform(X_test)

        self.decoder.fit(X_train_low, y_train)
        y_test_predict = self.decoder.predict(X_test_low)
        y_train_predict = self.decoder.predict(X_train_low)

        # Shuffle baseline: train a fresh estimator on permuted features.
        X_train_shuffle = np.random.permutation(X_train_low.flatten()).reshape(X_train_low.shape)
        shuffled_decoder = clone(self.decoder)
        shuffled_decoder.fit(X_train_shuffle, y_train)
        y_shuffle = shuffled_decoder.predict(X_test_low)

        accuracy_train = np.mean(y_train_predict == y_train)
        accuracy_test = np.mean(y_test_predict == y_test)
        chance = np.mean(y_shuffle == y_test)
        return accuracy_train, accuracy_test, chance

    def decode(self, test_size=0.3, n_repeats=50, n_time_bin=None):
        if self.X_to_use is None or self.y is None:
            raise ValueError("Set X_to_use and y before calling decode().")

        if n_time_bin is not None:
            assert isinstance(n_time_bin, int) and 0 <= n_time_bin < len(self.X_to_use), (
                "n_time_bin should be an integer between 0 and the number of time bins - 1"
            )

        all_test_accuracy = []
        all_train_accuracy = []
        all_chance = []

        if n_time_bin is None:
            print("We will build a decoder for each time bin")
            for repeat in range(n_repeats):
                print(f"Start repeat {repeat + 1}")
                for n_bin in range(len(self.X_to_use)):
                    acc_train, acc_test, chance = self._evaluate_split(
                        self.X_to_use[n_bin], self.y, test_size
                    )
                    all_train_accuracy.append(acc_train)
                    all_test_accuracy.append(acc_test)
                    all_chance.append(chance)

            self.test_accuracy = np.array(all_test_accuracy).reshape(n_repeats, -1)
            self.train_accuracy = np.array(all_train_accuracy).reshape(n_repeats, -1)
            self.chance = np.array(all_chance).reshape(n_repeats, -1)
        else:
            for repeat in range(n_repeats):
                print(f"Start repeat {repeat + 1}")
                acc_train, acc_test, chance = self._evaluate_split(
                    self.X_to_use[n_time_bin], self.y, test_size
                )
                all_train_accuracy.append(acc_train)
                all_test_accuracy.append(acc_test)
                all_chance.append(chance)

            self.test_accuracy = np.array(all_test_accuracy)
            self.train_accuracy = np.array(all_train_accuracy)
            self.chance = np.array(all_chance)

        self.mean_test_accuracy = np.mean(self.test_accuracy, 0)
        self.mean_train_accuracy = np.mean(self.train_accuracy, 0)
        self.mean_chance = np.mean(self.chance, 0)

        self.std_test_accuracy = np.std(self.test_accuracy, 0)
        self.std_train_accuracy = np.std(self.train_accuracy, 0)
        self.std_chance = np.std(self.chance, 0)
