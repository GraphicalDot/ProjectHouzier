#!/usr/bin/env ipython
#-*- coding: utf-8 -*-

import numpy as np
import Levenshtein 
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
import requests
import time


class KernelKMeans(BaseEstimator, ClusterMixin):
        """
        Kernel K-means
    
        Reference
        ---------
        Kernel k-means, Spectral Clustering and Normalized Cuts.
        Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
        KDD 2004.
        """

        def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None, kernel="linear", 
                gamma=None, degree=3, coef0=1, kernel_params=None, verbose=0):
                self.n_clusters = n_clusters
                self.max_iter = max_iter
                self.tol = tol
                self.random_state = random_state
                self.kernel = kernel
                self.gamma = gamma
                self.degree = degree
                self.coef0 = coef0
                self.kernel_params = kernel_params
                self.verbose = verbose
        

        def _get_kernel(self, X, Y=None):
                if Y is None:
                        K = self._L_kernels(X, X)
                else:
                        K = self._L_kernels(X, Y)
                return K

    
        def _L_kernels(self, X, Y):
                n_x, n_y = len(X), len(Y)
                # Calculate kernel for each element in X and Y.
                K = np.zeros((n_x, n_y), dtype='float')
                for i in range(n_x):
                        start = 0
                        if X is Y:
                                start = i
                                for j in range(start, n_y):
                                # Kernel assumed to be symmetric.
                                        K[i][j] = Levenshtein.distance(X[i],Y[j])
                                        if X is Y:
                                            K[j][i] = K[i][j]
                return K    


        def fit(self, X, y=None, sample_weight=None):
                n_samples = len(X)

                K = self._get_kernel(X)

                sw = sample_weight if sample_weight else np.ones(n_samples)
                self.sample_weight_ = sw

                rs = check_random_state(self.random_state)
                self.labels_ = rs.randint(self.n_clusters, size=n_samples)

                dist = np.zeros((n_samples, self.n_clusters))
                self.within_distances_ = np.zeros(self.n_clusters)

                for it in xrange(self.max_iter):
                        dist.fill(0)
                        self._compute_dist(K, dist, self.within_distances_, update_within=True)
                        labels_old = self.labels_
                        self.labels_ = dist.argmin(axis=1)

                        # Compute the number of samples whose cluster did not change 
                        # since last iteration.
                        n_same = np.sum((self.labels_ - labels_old) == 0)
                        if 1 - float(n_same) / n_samples < self.tol:
                                if self.verbose:
                                        print "Converged at iteration", it + 1
                        break

                self.X_fit_ = X

                return self

        def _compute_dist(self, K, dist, within_distances, update_within):
                """Compute a n_samples x n_clusters distance matrix using the 
                kernel trick."""
                sw = self.sample_weight_

                for j in xrange(self.n_clusters):
                        mask = self.labels_ == j

                        if np.sum(mask) == 0:
                                raise ValueError("Empty cluster found, try smaller n_cluster.")

                        denom = sw[mask].sum()
                        denomsq = denom * denom

                if update_within:
                        KK = K[mask][:, mask]  # K[mask, mask] does not work.
                        dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                        within_distances[j] = dist_j
                        dist[:, j] += dist_j
                else:
                        dist[:, j] += within_distances[j]

                dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

        def predict(self, X):
                K = self._get_kernel(X, self.X_fit_)
                n_samples = len(X)
                dist = np.zeros((n_samples, self.n_clusters))
                self._compute_dist(K, dist, self.within_distances_, update_within=False)
                return dist.argmin(axis=1)


if __name__ == '__main__':
        X = ['catch','liver','cat','katch','lever','cetch','level']
        y = ['catch']
        km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1)
        payload = {"eatery_id": "301489", "category":"food", "total_noun_phrases": 15, 
                "word_tokenization_algorithm": 'punkt_n_treebank', "pos_tagging_algorithm": "hunpos_pos_tagger",}


        start = time.time()
        r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
       
        X =  __result = [__dict.get("name").replace(" ", "") for __dict in r.json()["result"]]
        print km.fit_predict(X)
        print "Whole time taken is %s"%(time.time() - start)
