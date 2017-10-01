from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.model_selection import KFold
from datetime import timedelta
from collections import defaultdict
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/clean.csv')
df['saledate_converted'] = pd.to_datetime(df['saledate_converted'])
df = pd.get_dummies(df, columns = ['ProductGroup','Enclosure'])

'''
Create classes for gridsearch
'''
class ComputeNearestMean(BaseEstimator, TransformerMixin):
    """Compute a mean price for similar vehicles.
    """
    def __init__(self, window=5):
        self.window = window

    def get_params(self, **kwargs):
        return {'window': self.window}

    def fit(self, X,y):
        X = X.sort_values(by=['saledate_converted'])
        g = X.groupby('ModelID')['SalePrice']
        m = g.apply(lambda x: x.rolling(self.window).agg([np.mean]))

        ids = X[['saledate_converted', 'ModelID', 'SalesID']]
        z = pd.concat([m, ids], axis=1)
        z['saledate_converted'] = z.saledate_converted + timedelta(1)

        # Some days will have more than 1 transaction for a particular model,
        # take the last mean (which has most info)
        z = z.drop('SalesID', axis=1)
        groups = ['ModelID', 'saledate_converted']
        self.averages = z.groupby(groups).apply(lambda x: x.tail(1))

        # This is kinda unsatisfactory, but at least ensures
        # we can always make predictions
        self.default_mean = X.SalePrice.mean()
        return self

    def transform(self, X):
        near_price = pd.merge(self.averages, X, how='outer',
                              on=['ModelID', 'saledate_converted'])
        nxcols = ['ModelID', 'saledate_converted']
        near_price = near_price.set_index(nxcols).sort_index()
        g = near_price['mean'].groupby(level=0)
        filled_means = g.transform(lambda x: x.fillna(method='ffill'))
        near_price['filled_mean_price'] = filled_means
        near_price = near_price[near_price['SalesID'].notnull()]
        missing_mean = near_price.filled_mean_price.isnull()
        near_price['no_recent_transactions'] = missing_mean
        near_price['filled_mean_price'].fillna(self.default_mean, inplace=True)
        return near_price
class ColumnFilter(BaseEstimator, TransformerMixin):
    """Only use the following columns.
    """

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        columns = ['YearMade', 'year_change', 'equipment_age', 'filled_mean_price', 'no_recent_transactions',
        'ProductGroup_BL', 'ProductGroup_MG', 'ProductGroup_SSL',
        'ProductGroup_TEX', 'ProductGroup_TTT', 'ProductGroup_WL',
        'Enclosure_EROPS', 'Enclosure_EROPS AC', 'Enclosure_EROPS w AC',
        'Enclosure_NO ROPS', 'Enclosure_None or Unspecified',
        'Enclosure_OROPS']
        #'ProductGroup', 'Enclosure'
        X = X.set_index('SalesID')[columns].sort_index()
        #return pd.get_dummies(X, drop_first = True).values.astype(float)
        return X
def mean_squared_log_error(y_true, y_pred):
    y_pred = y_pred.clip(min=0)
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))

alphas = np.logspace(-4, -0.5, 30)
l_reg = {}
l_reg['model'] = 'Lasso'
l_reg['pipe'] = Pipeline([('CNM',ComputeNearestMean()), ('CF', ColumnFilter()),('scaler', StandardScaler()), ('Lasso', Lasso())])
l_reg['param'] = {'CNM__window':[5,10],
'Lasso__alpha': alphas,
'Lasso__normalize':[True],
'Lasso__max_iter' : [10000]}

random_f = {}
random_f['model'] = "RandomForest"
random_f['pipe'] = Pipeline([('CNM',ComputeNearestMean()),('CF', ColumnFilter()), ('rf',RandomForestRegressor())])
random_f['param'] = dict(
rf__n_estimators = [50, 100],
rf__max_features = ["sqrt"]
)

boosted_t = {}
boosted_t['model'] = "GradientBoost"
boosted_t['pipe'] = Pipeline([('CNM',ComputeNearestMean()),('CF', ColumnFilter()),('bt', GradientBoostingRegressor())])
boosted_t['param'] = dict(
CNM__window=[5, 10],
bt__n_estimators = [50,100],
bt__learning_rate = [0.05, 0.1, 0.5],
)

ada_boost = {}
ada_boost = {}
ada_boost['model'] = "AdaBoost"
ada_boost['pipe'] = Pipeline([('CNM',ComputeNearestMean()), ('CF', ColumnFilter()),('Ada', AdaBoostRegressor())])
ada_boost['param'] = dict(
CNM__window=[5, 10],
Ada__learning_rate = [0.5, 1, 1.5]
)


'''
Run gridsearch to find best models for lasso, Random forest, boosted trees, and adaboost
'''
y = df.SalePrice.values
X = df
models = defaultdict(list)

for model in [l_reg, random_f, boosted_t, ada_boost]:
    models['names'].append(model['model'])
    gscv = GridSearchCV(model['pipe'], model['param'],
    scoring = make_scorer(mean_squared_log_error,greater_is_better=False),cv = 10, n_jobs =-1)
    gscv.fit(X, y)
    print "1"
    models['estimators'].append(gscv.best_estimator_)
    models['scores'].append(gscv.best_score_)
print models
