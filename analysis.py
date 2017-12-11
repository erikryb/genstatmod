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
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from itertools import chain, combinations, count
import statsmodels.api as sm
from sklearn.model_selection import KFold
from math import sqrt

def prepareFrmgham2():
    dataset = pd.read_csv('frmgham2.csv')
    dataset = dataset[dataset.PERIOD == 1]
    #dataset = dataset[dataset.DEATH == 1]
    #dataset = dataset[dataset.PERIOD == 1]
    print dataset
    #X = dataset.ix[:,['SEX', 'AGE', 'SYSBP', 'DIABP',
    #X = dataset.ix[:,['SEX', 'AGE', 'SYSBP', 'DIABP',
    X = dataset.ix[:,['SEX', 'AGE',
        'CIGPDAY', 'BMI', 'DIABETES',
        'GLUCOSE', 'educ', 'TOTCHOL']]
    #X = dataset.ix[:,['SEX', 'AGE', 'BMI']]
    
    names = list(X)

    #for i1 in range(len(names)):
    #    for i2 in range(i1,len(names)):
    #        n1 = names[i1]
    #        n2 = names[i2]
    #        print n1,n2
    #        X[n1+'x'+n2] = X[n1] * X[n2]
    
    names = list(X)

    y = dataset.ix[:,'SYSBP']
    print X

    print X.ix[:5,:]

    c = pd.concat([X,y], axis=1)
    withoutNaN = np.array(c.dropna(axis=0, how='any'))
    np.random.seed(1)
    np.random.shuffle(withoutNaN)

    N = withoutNaN.shape[0]
    trainN = int(N*0.8)
    testN = N - trainN

    print trainN,testN,N

    trainX = scale(withoutNaN[:trainN,:-1])
    trainY = withoutNaN[:trainN,-1]

    testX = scale(withoutNaN[trainN:,:-1])
    testY = withoutNaN[trainN:,-1]

    P = trainX.shape[1]
    print N
    return names,trainX,trainY,testX,testY,trainN,P

def prepareProstate():
    dataset = pd.read_csv('prostate.csv', delim_whitespace=True)
    names = list(dataset)
    idxTrain = dataset[dataset.train == 'T'].index.values
    idxTest = dataset[dataset.train == 'F'].index.values
    #X = scale(np.array(dataset.ix[:,0:8]))
    #y = scale(np.array(dataset.ix[:,8]))
    X = np.array(dataset.ix[:,0:8])
    y = np.array(dataset.ix[:,8])

    trainX = X[idxTrain-1,:]
    trainY = y[idxTrain-1]
    
    testX = X[idxTest-1,:]
    testY = y[idxTest-1]

    N = trainX.shape[0]
    P = trainX.shape[1]
    return names,trainX,trainY,testX,testY,N,P

#names,trainX,trainY,testX,testY,N,P = prepareProstate()
names,trainX,trainY,testX,testY,N,P = prepareFrmgham2()

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
        ax1.set_ylabel('Koeffisienter')
        ax1.set_title('Lasso-regresjon')
        ax1.axis('tight')

        ax2.set_ylabel('R^2')
        ax2.plot(*zip(*[(t, s) for (_,t,s) in scores]))
        ax2.set_xlabel('|coef| / max|coef|')

        plt.savefig('lasso.png')

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
        plt.xlabel('Antall faktorer i delmengden')
        plt.ylabel('R^2')
        plt.title("Beste delmengde-utvalg")
        plt.savefig('best_subset.png')
        #plt.show()
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

        ax2.set_xlabel('Antall prinsipalkomponenter')
        ax1.set_ylabel('Varians forklart')
        ax1.axis('tight')

        ax2.set_ylabel('R^2')
        ax2.plot(range(0,n+1), [s for (_,s) in scores])

        plt.savefig("pcr.png")

    return (pcaModel, zip(range(0,n+1), [m for (m,_) in scores]))

nsplits = 10

def best_subsetCV():
    errors = [[] for i in range(0,P+1)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        scalerX = StandardScaler().fit(trainX[train_index,:])
        scalerY = StandardScaler().fit(trainY[train_index].reshape(-1,1))
        Xs = scalerX.transform(trainX[train_index,:])
        ys = scalerY.transform(trainY[train_index].reshape(-1,1))[:,0]
        Xtest = scalerX.transform(trainX[test_index,:])
        Ytest = scalerY.transform(trainY[test_index].reshape(-1,1))[:,0]
        
        models = [(s,m) for (_,(s,m,_)) in best_subset(Xs,ys)]
        X = Xtest
        for (s,m) in models:
            if s == ():
                pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
            else:
                pred = m.predict(X[:,s])
            err = pred - Ytest
            L = err**2
            errors[len(s)].append(list(L))
    return range(0,P+1), errors

def best_subsetTest(Xtrain,Ytrain,Xpred,Ypred,k):
    scalerX = StandardScaler().fit(Xtrain)
    scalerY = StandardScaler().fit(Ytrain.reshape(-1,1))
    Xs = scalerX.transform(Xtrain)
    ys = scalerY.transform(Ytrain.reshape(-1,1))[:,0]
    Xtest = scalerX.transform(Xpred)
    Ytest = scalerY.transform(Ypred.reshape(-1,1))[:,0]
    
    (_,(s,m,_)) = best_subset(Xs,ys)[k]
    X = Xtest
    if s == ():
        pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
    else:
        pred = m.predict(X[:,s])
    err = pred - Ytest
    L = err**2
    mErr = sum(L)/len(L)
    ste = sqrt(sum((err - mErr)**2)/len(L))/sqrt(Xtest.shape[0])
    return ([names[i] for i in s], list(m.coef_), mErr, ste)

def pcrCV():
    errors = [[] for i in range(0,P+1)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        scalerX = StandardScaler().fit(trainX[train_index,:])
        scalerY = StandardScaler().fit(trainY[train_index].reshape(-1,1))
        Xs = scalerX.transform(trainX[train_index,:])
        ys = scalerY.transform(trainY[train_index].reshape(-1,1))[:,0]
        Xtest = scalerX.transform(trainX[test_index,:])
        Ytest = scalerY.transform(trainY[test_index].reshape(-1,1))[:,0]
        pcaModel, models = pcr(Xs,ys)
        X = pcaModel.transform(Xtest)
        for (i,m) in models:
            if i == 0:
                pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
            else:
                pred = m.predict(X[:,:i])
            err = pred - Ytest
            L = err**2
            errors[i].append(list(L))
    return range(0,P+1), errors

def pcrTest(Xtrain,Ytrain,Xpred,Ypred,k):
    scalerX = StandardScaler().fit(Xtrain)
    scalerY = StandardScaler().fit(Ytrain.reshape(-1,1))
    Xs = scalerX.transform(Xtrain)
    ys = scalerY.transform(Ytrain.reshape(-1,1))[:,0]
    Xtest = scalerX.transform(Xpred)
    Ytest = scalerY.transform(Ypred.reshape(-1,1))[:,0]
    pcaModel, models = pcr(Xs,ys)
    X = pcaModel.transform(Xtest)
    (_,m) = models[k]
    if k == 0:
        pred = m.predict(np.ones_like(X[:,0]).reshape(-1,1))
    else:
        pred = m.predict(X[:,:k])
    err = pred - Ytest
    L = err**2
    #print pcaModel.components_.T[:,:k]
    coefs = np.dot(pcaModel.components_.T[:,:k],m.coef_)
    mErr = sum(L)/len(L)
    ste = sqrt(sum((err - mErr)**2)/len(L))/sqrt(len(L))
    return (list(coefs), mErr, ste)

def lassoCV():
    xs = []
    errors = [[] for i in range(0,11)]
    kf = KFold(n_splits=nsplits)
    for train_index, test_index in kf.split(trainX):
        scalerX = StandardScaler().fit(trainX[train_index,:])
        scalerY = StandardScaler().fit(trainY[train_index].reshape(-1,1))
        Xs = scalerX.transform(trainX[train_index,:])
        ys = scalerY.transform(trainY[train_index].reshape(-1,1))[:,0]
        Xtest = scalerX.transform(trainX[test_index,:])
        Ytest = scalerY.transform(trainY[test_index].reshape(-1,1))[:,0]

        ts, models = lasso(Xs,ys)
        xs = ts
        for (i,(t,c)) in enumerate(zip(ts,models)):
            pred = np.dot(Xtest,c.T)
            err = pred - Ytest
            L = err**2
            errors[i].append(list(L))
    return xs, errors

def lassoTest(Xtrain,Ytrain,Xpred,Ypred,k):
    scalerX = StandardScaler().fit(Xtrain)
    scalerY = StandardScaler().fit(Ytrain.reshape(-1,1))
    Xs = scalerX.transform(Xtrain)
    ys = scalerY.transform(Ytrain.reshape(-1,1))[:,0]
    Xtest = scalerX.transform(Xpred)
    Ytest = scalerY.transform(Ypred.reshape(-1,1))[:,0]
    _, models = lasso(Xs,ys)
    c = models[k]
    pred = np.dot(Xtest,c.T)
    err = pred - Ytest
    L = err**2
    mErr = sum(L)/len(L)
    ste = sqrt(sum((err - mErr)**2)/len(L))/sqrt(Xtest.shape[0])
    return (list(c), mErr, ste)

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
        #std = sqrt(np.sum(np.dot(means-m,np.transpose(means-m)))/(N-1))/(sqrt(len(means)-1))
        std = np.std(means)/(sqrt(len(i)))
        all_stds.append(std)
    return (all_means,all_stds)

def oneStandardError(xs,means,stds):
    thresh = min([m+e for (m,e) in zip(means,stds)])
    for (x,m) in zip(xs,means):
        if m <= thresh:
            return (thresh, (x,m))

def plotEstimatedError(xs,means,stds,xname,filename):
    thresh, (x0,m0) = oneStandardError(xs,means,stds)
    
    fig, ax = plt.subplots()
    
    ax.plot([xs[0],xs[-1]], [thresh,thresh], ls='dashed')
    ax.set_xlabel(xname)
    ax.set_ylabel('CV Error')
    
    ax.plot(xs,means)
    ax.errorbar(xs,means, yerr=stds, ecolor="blue", capthick=1, fmt='o')
    ax.plot(x0, m0, 'o', color='r', markersize=10, fillstyle="none")
    plt.savefig(filename+'.png')

def makeLinearFit(Xtrain,Ytrain,Xpred,Ypred):
    scalerX = StandardScaler().fit(Xtrain)
    scalerY = StandardScaler().fit(Ytrain.reshape(-1,1))
    Xs = scalerX.transform(Xtrain)
    ys = scalerY.transform(Ytrain.reshape(-1,1))[:,0]
    Xtest = scalerX.transform(Xpred)
    Ytest = scalerY.transform(Ypred.reshape(-1,1))[:,0]
    model = LinearRegression().fit(Xs,ys)
    pred = model.predict(Xtest)
    err = pred - Ytest
    L = err**2
    #print pcaModel.components_.T[:,:k]
    mErr = sum(L)/len(L)
    ste = sqrt(sum((err - mErr)**2)/len(L))/sqrt(len(L))
    return (model.coef_, mErr, ste)

print(makeLinearFit(trainX,trainY,testX,testY))

#best_subset(trainX,trainY,makeplot=True)
#lasso(scale(trainX),scale(trainY),makeplot=True)
#pcr(scale(trainX),scale(trainY),makeplot=True)

xs, errors = best_subsetCV()
means,stds = getMeansStds(errors)
plotEstimatedError(xs, means, stds,"Antall faktorer i delmengden","best_subset_CV")
(_,(k,_)) = oneStandardError(xs,means,stds)
print best_subsetTest(trainX,trainY,testX,testY,k)

xs, errors = pcrCV()
means,stds = getMeansStds(errors)
plotEstimatedError(xs, means, stds,"Antall prinsipalkomponenter","pcr_CV")
(_,(k,_)) = oneStandardError(xs,means,stds)
print pcrTest(trainX,trainY,testX,testY,k)
#(c,_,_) = pcrTest(trainX,trainY,testX,testY,k)

#print sorted(zip(list(abs(np.array(c))),names,c))

xs, errors = lassoCV()
means,stds = getMeansStds(errors)
plotEstimatedError(xs, means, stds,'|coef| / max|coef|', "lasso_CV")
(_,(k,_)) = oneStandardError(range(0,11),means,stds)
print lassoTest(trainX,trainY,testX,testY,k)
#(c,_,_) = lassoTest(trainX,trainY,testX,testY,k)

#print sorted(zip(list(abs(np.array(c))),names,c))
