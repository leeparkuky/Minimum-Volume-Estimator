import numpy as np
import pandas as pd
from numpy.linalg import inv

class MinimumVolumeEllipsoid:
    def __init__(self, df, k, level):
        self.X = df
        self.X_after = df.copy()
        self.n, self.p = df.shape
        self.h = int(np.floor((self.n + self.p + 1 )/2))
        self.crit = {}
        self.xjs = {}
        self.covs = {}
        self.md_sqs = {}
        self.k = k
        self.level = level
        self.gen_md_sq()
        
        
        
        
    def __enter__(self):
        return self
    
    
    
    
    def __exit__(self, *a):
        pass
        
        
        
        
    def sampling(self):
        index = np.random.choice(np.arange(self.n), size = self.p+1, replace = False)
        self.index = index
    
    
    
    
    def gen_subset(self):
        self.sampling()
        self.subset = self.X.iloc[self.index, :]
        return(self.subset)
    
    
    
    
    def gen_md_sq(self):
        from numpy.linalg import inv
        from sklearn.neighbors import DistanceMetric
        X_bar = np.mean(self.X, axis = 0)
        cov = np.cov(self.X.T)
        dist = DistanceMetric.get_metric('mahalanobis', V =  cov)
        cov_inv = inv(cov)
        MD_sq = np.array(list(map(lambda i: ((self.X.iloc[i,:]- X_bar).T).dot(cov_inv).dot(self.X.iloc[i,:]- X_bar), range(self.n))))
        self.X_after['MD_sq'] = MD_sq
        
        
        
        
    def gen_mj_sq(self):
        from sklearn.neighbors import DistanceMetric
        X = self.gen_subset()
        X_bar = np.mean(X, axis = 0)
        self.xj = X_bar
        cov = np.cov(X.T)
        self.cov = cov
        det = np.linalg.det(cov)
        if det != 0:
            pass
        else:
            while det == 0:
                X = self.gen_subset()
                X_bar = np.mean(X, axis = 0)
                self.xj = X_bar
                cov = np.cov(X.T)
                self.cov = cov
                det = np.linalg.det(cov)
        N = self.subset.shape[0]
        dist = DistanceMetric.get_metric('mahalanobis', V =  cov)
        cov_inv = inv(cov)
        MD_sq = np.array(list(map(lambda i: ((self.X.iloc[i,:]- X_bar).T).dot(cov_inv).dot(self.X.iloc[i,:]- X_bar), range(self.n))))
        MD_sq.sort()
        md_sq = np.mean([MD_sq[self.h-1],MD_sq[self.h]])
        self.md_sq = md_sq
    
    
    
    
    def find_mj_sq(self):
        k = self.k
        for i in range(k):
            self.gen_mj_sq()
            self.md_sqs[i] = self.md_sq
            self.crit[i] = self.md_sq**self.p * np.linalg.det(self.cov)
            self.xjs[i] = self.xj
            self.covs[i] = self.cov
        crit = min(self.crit.values())
        crit_index = list(self.crit.values()).index(crit)
        xj = self.xjs[crit_index]
        cov = self.covs[crit_index]
        self.md_sq  = self.md_sqs[crit_index]
        self.xj = xj
        self.cov = cov
    
    
    
    
    def get_statistics(self):
        from scipy.stats import chi2
        self.find_mj_sq()
        chi_sq = chi2.ppf(.5, self.p)
        c = (1 + 15/(self.n - self.p))**2
        if self.n < 40:
            self.cov = self.cov / chi_sq * self.md_sq * c
        else:
            self.cov = self.cov / chi_sq * self.md_sq        
        return(self.xj, self.cov)
    
    
    
    
    def gen_rd(self):
        from sklearn.neighbors import DistanceMetric
        self.get_statistics()
        cov = self.cov
        X_bar = self.xj
        cov_inv = inv(cov)
        dist = DistanceMetric.get_metric('mahalanobis', V =  cov)
        MD_sq =  np.array(list(map(lambda i: ((self.X.iloc[i,:]- X_bar).T).dot(cov_inv).dot(self.X.iloc[i,:]- X_bar), range(self.n))))
        self.X_after['RD_sq'] = MD_sq
        return(self.X_after)
        
        
            
    def chisq(self):
        from scipy.stats import chi2
        self.chisq_value = chi2.ppf(self.level, self.p)
        return(self.chisq_value)
        
        
        
        
    def plot(self, filename):
        import matplotlib.pyplot as plt
        self.chisq()
        output = self.X_after
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        ax.scatter(x = np.arange(1, output.shape[0]+1), y = output.MD_sq, alpha = .3, color = 'orange', label = 'MD^2')
        ax.scatter(x = np.arange(1, output.shape[0]+1), y = output.RD_sq, alpha = 1, color = 'blue', label = 'RD^2')
        ax.hlines(y = self.chisq_value, xmin = 1, xmax =  output.shape[0]+1, color = 'red', linestyle = '--')
        ax.set_xticklabels("")
        plt.legend()
        plt.savefig(filename)