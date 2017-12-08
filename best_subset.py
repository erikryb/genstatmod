import numpy as np
import pandas as pd
import urllib
from itertools import chain, combinations
import statsmodels.api as sm
import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

dataset = pd.read_csv('frmgham2.csv')
#dataset.columns = "RANDID,SEX,TOTCHOL,AGE,SYSBP,DIABP,CURSMOKE,CIGPDAY,BMI,DIABETES,BPMEDS,HEARTRTE,GLUCOSE,educ,PREVCHD,PREVAP,PREVMI,PREVSTRK,PREVHYP,TIME,PERIOD,HDLC,LDLC,DEATH,ANGINA,HOSPMI,MI_FCHD,ANYCHD,STROKE,CVD,HYPERTEN,TIMEAP,TIMEMI,TIMEMIFC,TIMECHD,TIMESTRK,TIMECVD,TIMEDTH,TIMEHYP".split(',')

def best_subset(X, y):
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(xrange(n_features), k+1) for k in xrange(n_features))
    scores = []
    for subset in subsets:
        lin_reg = sm.OLS(y, sm.add_constant(X.iloc[:, list(subset)]), missing='drop').fit()
        score = lin_reg.rsquared
        scores.append((subset, score))
    return scores

#print dataset.head(5)

#X = dataset.ix[:,2:6]
#y = dataset.ix[:,14]

X = dataset.ix[:,2:12]
y = dataset.ix[:,"PREVHYP"]

#c = pd.concat([X,y], axis=1)
#withoutNaN = c.dropna(axis=0, how='any')
#X = withoutNaN.ix[:,:-1]
#y = withoutNaN.ix[:,-1]
#print X
#print y

scores = best_subset(X, y)

bestscores = dict()
for (sub,s) in scores:
    if len(sub) not in bestscores or bestscores[len(sub)] < s:
        bestscores[len(sub)] = s

#plt.figure()
plt.scatter(*zip(*[(len(sub), s) for (sub,s) in scores]))
plt.plot(*zip(*(bestscores.items())), color='red')
#label='Average across the folds', linewidth=2)
#plt.axvline(-np.log10(model.alpha_), linestyle='-', color='k',
#label='alpha CV')
#plt.legend()
plt.xlabel('Number of covariates')
plt.ylabel('R^2')
plt.title("Best subset selection")
plt.show()
