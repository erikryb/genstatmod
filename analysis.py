import numpy as np
import pandas as pd
import urllib
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.linear_model import LassoLars
from sklearn.linear_model import lars_path
from sklearn.preprocessing import scale
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from itertools import chain, combinations, count
import statsmodels.api as sm
from sklearn.model_selection import KFold
from math import sqrt

def prepareFrmgham2():
    dataset = pd.read_csv('frmgham2.csv')
    X = dataset.ix[:,1:9]
    y = dataset.ix[:,9]

    c = pd.concat([X,y], axis=1)
    withoutNaN = np.array(c.dropna(axis=0, how='any'))
    np.random.shuffle(withoutNaN)

    N = withoutNaN.shape[0]
    trainN = int(N*0.8)
    testN = N - trainN

    trainX = scale(withoutNaN[:trainN,:-1])
    trainY = withoutNaN[:trainN,-1]

    testX = scale(withoutNaN[trainN:,:-1])
    testY = withoutNaN[trainN:,-1]

    P = trainX.shape[1]

def prepareProstate():
    dataset = pd.read_csv('prostate.csv', delim_whitespace=True)
    idx = dataset[dataset.train == 'T'].index.values
    X = scale(np.array(dataset.ix[:,0:8]))
    y = scale(np.array(dataset.ix[:,8]))
    #X = np.array(dataset.ix[:,0:8])
    #y = np.array(dataset.ix[:,8])

    trainX = X[idx-1,:]
    trainY = y[idx-1]

    N = trainX.shape[0]
    P = trainX.shape[1]
    return trainX,trainY,N,P

trainX,trainY,N,P = prepareProstate()

def lasso(X,y,makeplot=False):
    (alphas,_,coefs) = lars_path(X,y, method="lasso")

    xx = np.sum(np.abs(coefs.T), axis=1)
    max_xx = xx[-1]
    xx /= max_xx
    
    coef_path_continuous = interpolate.interp1d(xx, coefs)

    scores = []
    t_vals_plot = [i*0.01 for i in range(0,101)]
    for t in t_vals_plot:
        c = coef_path_continuous(t).T
        fit = np.dot(X,(c.T))
        m = sum(y)/len(y)
        ss_tot = np.dot((y-m),(y-m).T)
        ss_res = np.dot((y-fit),(y-fit))
        r_squared = 1 - (ss_res/ss_tot)
        scores.append((c, t, r_squared))
    
    if makeplot:
        f, (ax1,ax2) = plt.subplots(2)
        ax1.plot(xx, coefs.T)
        ymin, ymax = ax1.get_ylim()

        ax1.vlines(xx, ymin, ymax, linestyle='dashed')
        ax1.set_ylabel('Coefficients')
        ax1.set_title('LASSO Path')
        ax1.axis('tight')

        ax2.set_ylabel('R^2')
        ax2.plot(*zip(*[(t, s) for (_,t,s) in scores]))
        ax2.set_xlabel('|coef| / max|coef|')

        plt.show()

    t_vals = [i*0.1 for i in range(0,11)]
    models = coef_path_continuous(t_vals).T
    return (t_vals, models)

def best_subset(X,y,makeplot=False):
    n_features = X.shape[1]
    subsets = [()] + list(chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features)))
    scores = []
    for subset in subsets:
        if subset == ():
            onesX = np.ones_like(X[:,0]).reshape(-1,1)
            model = LinearRegression().fit(onesX,y)
            score = model.score(onesX,y)
        else:
            model = LinearRegression().fit(X[:, list(subset)],y)
            score = model.score(X[:, list(subset)],y)
        scores.append((subset, model, score))
    
    bestscores = dict()
    for (sub,m,s) in scores:
        if len(sub) not in bestscores or bestscores[len(sub)][2] < s:
            bestscores[len(sub)] = (sub,m,s)
    
    if makeplot:
        plt.scatter(*zip(*[(len(sub), s) for (sub,_,s) in scores]))
        plt.plot(*zip(*([(n,s) for (n,(sub,m,s)) in bestscores.items()])), color='red')
        plt.xlabel('Number of covariates')
        plt.ylabel('R^2')
        plt.title("Best subset selection")
        plt.show()
    return bestscores.items()

def pcr(X,y,makeplot=False):
    pca = PCA()

    pcaModel = pca.fit(X)
    X_reduced = pcaModel.transform(X)

    variance_explained = list(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))

    scores = []
    n = X_reduced.shape[1]
    for i in range(0,n+1):
        if i == 0:
            onesX = np.ones_like(X[:,0]).reshape(-1,1)
            model = LinearRegression().fit(onesX,y)
            scores.append((model, model.score(onesX,y)))
        else:
            model = LinearRegression().fit(X_reduced[:,:i],y)
            scores.append((model, model.score(X_reduced[:,:i],y)))

    if makeplot:
        f, (ax1,ax2) = plt.subplots(2)

        ax1.plot(range(1,n+1),variance_explained)
        ymin, ymax = ax1.get_ylim()

        ax2.set_xlabel('Number of principal components')
        ax1.set_ylabel('Variance explained')
        ax1.axis('tight')

        ax2.set_ylabel('R^2')
        ax2.plot(range(1,n+1), [s for (_,s) in scores])

        plt.show()

    return (pcaModel, zip(range(0,n+1), [m for (m,_) in scores]))

nsplits = 10

def best_subsetCV():
    errors = [[] for i in range(0,P+1)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        models = [(s,m) for (_,(s,m,_)) in best_subset(trainX[train_index,:],trainY[train_index])]
        X = trainX[test_index,:]
        for (s,m) in models:
            if s == ():
                pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
            else:
                pred = m.predict(X[:,s])
            err = pred - trainY[test_index]
            L = err**2
            errors[len(s)].append(list(L))
    return range(0,P+1), errors

def pcrCV():
    errors = [[] for i in range(0,P+1)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        pcaModel, models = pcr(trainX[train_index,:],trainY[train_index])
        X = pcaModel.transform(trainX[test_index])
        for (i,m) in models:
            if i == 0:
                pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
            else:
                pred = m.predict(X[:,:i])
            err = pred - trainY[test_index]
            L = err**2
            errors[i].append(list(L))
    return range(0,P+1), errors

def lassoCV():
    xs = []
    errors = [[] for i in range(0,11)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        ts, models = lasso(trainX[train_index,:],trainY[train_index])
        xs = ts
        for (i,(t,c)) in enumerate(zip(ts,models)):
            pred = np.dot(trainX[test_index],c.T)
            err = pred - trainY[test_index]
            L = err**2
            errors[i].append(list(L))
    return xs, errors

def getMeansStds(errors):
    all_means = []
    all_stds = []
    for i in errors:
        means = []
        m = 0
        for fold in i:
            means.append(sum(fold)/len(fold))
            m += sum(fold)
        m = m/N
        means = np.array(means)
        #m = sum(means)/len(means)
        all_means.append(m)
        std = sqrt(np.sum(np.dot(means-m,np.transpose(means-m)))/(N))/(sqrt(len(means)))
        all_stds.append(std)
    return (all_means,all_stds)

def plotEstimatedError(xs,means,stds):
    fig, ax = plt.subplots()
    thresh = min([m+e for (m,e) in zip(means,stds)])
    ax.plot([xs[0],xs[-1]], [thresh,thresh], ls='dashed')

    for (x,m) in zip(xs,means):
        if m <= thresh:
            firstbelow = (x,m)
            break

    (x0,m0) = firstbelow
    ax.plot(xs,means)
    ax.errorbar(xs,means, yerr=stds, ecolor="blue", capthick=1, fmt='o')
    ax.plot(x0, m0, 'o', color='r', markersize=10, fillstyle="none")
    plt.show()

def makeLinearFit(X,y):
    model = LinearRegression().fit(X,y)
    return model.coef_

#print(makeLinearFit(trainX,trainY))

#xs, errors = best_subsetCV()
xs, errors = pcrCV()
#xs, errors = lassoCV()
means,stds = getMeansStds(errors)
plotEstimatedError(xs, means, stds)

#lasso(trainX,trainY,True)

#m = best_subset(trainX,trainY)
#lasso(trainX,trainY)
