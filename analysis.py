import numpy as np
import pandas as pd
import urllib
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoLars
from sklearn.linear_model import lars_path
from sklearn.preprocessing import scale
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from itertools import chain, combinations
import statsmodels.api as sm

dataset = pd.read_csv('frmgham2.csv')

X = dataset.ix[:,2:12]
y = dataset.ix[:,"PREVHYP"]

c = pd.concat([X,y], axis=1)
withoutNaN = c.dropna(axis=0, how='any')
X = scale(withoutNaN.ix[:,:-1])
#X = withoutNaN.ix[:,:-1]
y = withoutNaN.ix[:,-1]

#coefs = [[] for _ in xrange(X.shape[1])]
#alpha_vals = [x*0.01 for x in range(1,41)]
#for a in alpha_vals:
#    model=Lasso(alpha=a).fit(X,y)
#    coef = list(model.coef_)
#    for i in range(len(coefs)):
#        coefs[i].append(coef[i])

#for c in coefs:
#    plt.plot(alpha_vals, c)
#plt.show()

def lasso(X,y):
    (_,_,coefs) = lars_path(X,y, method="lasso")

    xx = np.sum(np.abs(coefs.T), axis=1)
    max_xx = xx[-1]
    xx /= max_xx

    f, (ax1,ax2) = plt.subplots(2)

    ax1.plot(xx, coefs.T)
    ymin, ymax = ax1.get_ylim()

    ax1.vlines(xx, ymin, ymax, linestyle='dashed')
    ax1.set_ylabel('Coefficients')
    ax1.set_title('LASSO Path')
    ax1.axis('tight')

    scores = []
    alpha_vals = [(0.001/10) * i for i in range(0,101)]
    for a in alpha_vals:
        model = LassoLars(alpha=a).fit(X,y)
        scores.append((np.sum(np.abs(model.coef_)), model.score(X,y)))

    scores.reverse()
    #print(scores)

    ax2.set_ylabel('R^2')
    ax2.plot(*zip(*[(a/max_xx, s) for (a,s) in scores]))
    ax2.set_xlabel('|coef| / max|coef|')

    plt.show()

def best_subset(X,y):
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features))
    scores = []
    for subset in subsets:
        lin_reg = sm.OLS(y, sm.add_constant(X[:, list(subset)]), missing='drop').fit()
        score = lin_reg.rsquared
        scores.append((subset, score))
    
    bestscores = dict()
    for (sub,s) in scores:
        if len(sub) not in bestscores or bestscores[len(sub)] < s:
            bestscores[len(sub)] = s
    
    plt.scatter(*zip(*[(len(sub), s) for (sub,s) in scores]))
    plt.plot(*zip(*(bestscores.items())), color='red')
    plt.xlabel('Number of covariates')
    plt.ylabel('R^2')
    plt.title("Best subset selection")
    plt.show()

def pcr(X,y):
    pca = PCA()

    dataset = pd.read_csv('frmgham2.csv')

    X = dataset.ix[:,2:12]
    y = dataset.ix[:,"PREVHYP"]

    c = pd.concat([X,y], axis=1)
    withoutNaN = c.dropna(axis=0, how='any')
    X = scale(withoutNaN.ix[:,:-1])
    #X = withoutNaN.ix[:,:-1]
    y = withoutNaN.ix[:,-1]

    X_reduced = pca.fit_transform(X)

    variance_explained = list(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))

    scores = []
    n = X_reduced.shape[1]
    for i in range(1,n+1):
        model = LinearRegression().fit(X_reduced[:,:i],y)
        scores.append(model.score(X_reduced[:,:i],y))

    f, (ax1,ax2) = plt.subplots(2)

    print(scores)

    print(list(enumerate(variance_explained)))

    ax1.plot(range(1,n+1),variance_explained)
    ymin, ymax = ax1.get_ylim()

    ax2.set_xlabel('Number of principal components')
    ax1.set_ylabel('Variance explained')
    ax1.axis('tight')

    ax2.set_ylabel('R^2')
    ax2.plot(range(1,n+1), scores)
    #ax2.set_xlabel('|coef| / max|coef|')

    plt.show()

#lasso(X,y)
#pcr(X,y)
best_subset(X,y)
