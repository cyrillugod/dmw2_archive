#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics import confusion_matrix
from sklearn.base import clone


# In[2]:


def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
        
    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    X = np.asarray(X)
    return (np.sum([(dist(x, c) ** 2) / (2 * len(X[np.where(y==i)[0]])) 
                    for i, c in enumerate(centroids) 
                        for x in X[np.where(y==i)[0]]]))


# In[4]:


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference 
        realizations
    random_state : int, default=None
        Determines random number generation for realizations
        
    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    rng = np.random.default_rng(random_state)
    log_wki = []
    kmeans_b = clusterer
    for i in range(b):
        X_rand = rng.uniform(X.min(axis=0), X.max(axis=0), X.shape)
        y_rand = kmeans_b.fit_predict(X_rand)
        centroids_b = kmeans_b.cluster_centers_
        log_wki.append(np.log(pooled_within_ssd(X_rand, y_rand,
                                                centroids_b, dist)))
    gap = ((np.mean(log_wki)) 
                - np.log(pooled_within_ssd(X, y, centroids, dist)))
    return gap, np.std(log_wki)


# In[6]:


def purity(y_true, y_pred):
    """Compute the class purity
    
    Parameters
    ----------
    y_true : array
        List of ground-truth labels
    y_pred : array
        Cluster labels
        
    Returns
    -------
    purity : float
        Class purity
    """
    return confusion_matrix(y_true, y_pred).max(axis=0).sum() / len(y_true)


# In[7]:


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """
    Accepts the design matrix, clustering object, k values, and an optional
    actual labels and returns a dictionary of cluster labels and centers, and
    internal and external validation criteria.
    """
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    for k in range(k_start, k_stop+1):
        
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k)
        y = clusterer_k.fit_predict(X)
        gs = gap_statistic(X, y, clusterer_k.cluster_centers_, 
                                 euclidean, 5, 
                                 clone(clusterer).set_params(n_clusters=k), 
                                 random_state=1337)
        gss.append(gs[0])
        gssds.append(gs[1])
        centers.append(clusterer_k.cluster_centers_)
        inertias.append(clusterer_k.inertia_)
        ys.append(y)
        chs.append(calinski_harabasz_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
        scs.append(silhouette_score(X, y))
        
        if actual is not None:
            ps.append(purity(actual, y))
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y))
            
    if actual is not None:
        cluster_dict = {
            'ys': ys,
            'centers': centers,
            'inertias': inertias,
            'chs': chs,
            'scs': scs,
            'dbs': dbs,
            'gss': gss,
            'gssds': gssds,
            'ps': ps,
            'amis': amis,
            'ars': ars
        }
    else:
        cluster_dict = {
            'ys': ys,
            'centers': centers,
            'inertias': inertias,
            'chs': chs,
            'scs': scs,
            'dbs': dbs,
            'gss': gss,
            'gssds': gssds
    }
    return cluster_dict


# In[8]:


def plot_clusters(X, ys, centers, transformer):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y,cs in zip(range(2, k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax


# In[9]:


def plot_internal(inertias, chs, scs, dbs, gss, gssds):
    """Plot internal validation values"""
    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt='-go', label='Gap statistic')
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.plot(ks, dbs, '-gs', label='DB')
    ax2.set_ylabel('Gap statistic/Silhouette/DB')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax


# In[10]:


def plot_external(ps, amis, ars):
    """Plot external validation values"""
    fig, ax = plt.subplots()
    ks = np.arange(2, len(ps)+2)
    ax.plot(ks, ps, '-o', label='PS')
    ax.plot(ks, amis, '-ro', label='AMI')
    ax.plot(ks, ars, '-go', label='AR')
    ax.set_xlabel('$k$')
    ax.set_ylabel('PS/AMI/AR')
    ax.legend()
    return ax


# In[11]:


def gap_statistic_kmedoids(X, y, centroids, b):
    """Compute the gap statistic for a k-medoids clusterer
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    b : int
        Number of realizations for the reference distribution
        
    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    np.random.seed(1337)
    X = np.asarray(X)
    log_wki = []
    medoids = []
    
    # convert centroids to medoids
    for i, c in enumerate(centroids):
        dist_list = []
        for x in X[y==i]:
            dist_list.append(euclidean(x, c))
        medoids.append(X[y==i][np.argmin(dist_list)])
    
    # medoids of bootstrap
    for _ in range(b):
        X_rand = np.random.uniform(low=np.min(X, axis=0),
                                     high=np.max(X, axis=0),
                                     size=X.shape)
        kmo_b = kmedoids(X_rand, np.arange(len(centroids)), ccore=True)
        kmo_b.process()
        y_predict = np.zeros(len(X), dtype=int)
        clusters = kmo_b.get_clusters()
        for cluster, point in enumerate(clusters):
            y_predict[point] = cluster
        medoids_b = X_rand[kmo_b.get_medoids(), :]
        log_wki.append(np.log(pooled_within_ssd(X_rand, y_predict,
                                                medoids_b, euclidean)))
    medoids = np.asarray(medoids)
    gap = ((np.mean(log_wki)) 
                - np.log(pooled_within_ssd(X, y, centroids, euclidean)))
    return gap, np.std(log_wki)


# In[12]:


def cluster_range_kmedoids(X, k_start, k_stop, actual=None):
    """
    Accepts the design matrix, clustering object, k values, and an optional
    actual labels and returns a dictionary of cluster labels and centers, and
    internal and external validation criteria.
    """
    ys = []
    cs = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    X = np.asarray(X)
    for k in range(k_start, k_stop+1):
        kmo = kmedoids(X, np.arange(k), ccore=True)
        
        kmo.process()
        y = np.zeros(len(X), dtype=int)
        clusters = kmo.get_clusters()
        for cluster, point in enumerate(clusters):
            y[point] = cluster
        centers = X[kmo.get_medoids(), :]
        
        sse = np.sum([euclidean(x, c) ** 2 for i, c in enumerate(centers)
                        for x in X[y==i]])

        gs = gap_statistic_kmedoids(X, y, centers, 5)
        gss.append(gs[0])
        gssds.append(gs[1])
        cs.append(centers)
        inertias.append(sse)
        ys.append(y)
        chs.append(calinski_harabasz_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
        scs.append(silhouette_score(X, y))
        
        if actual is not None:
            ps.append(purity(actual, y))
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y))
    
    if actual is not None:
        cluster_dict = {
            'ys': ys,
            'centers': cs,
            'inertias': inertias,
            'chs': chs,
            'scs': scs,
            'dbs': dbs,
            'gss': gss,
            'gssds': gssds,
            'ps': ps,
            'amis': amis,
            'ars': ars
        }
    else:
        cluster_dict = {
            'ys': ys,
            'centers': cs,
            'inertias': inertias,
            'chs': chs,
            'scs': scs,
            'dbs': dbs,
            'gss': gss,
            'gssds': gssds
    }
    return cluster_dict

