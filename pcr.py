import numpy as np
import pandas as pd
import urllib
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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
