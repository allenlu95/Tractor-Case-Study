from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from clean_df import X, y

df = df.drop(['ModelID','MachineID','SalesID','saledate_converted'], axis = 1)
y = df.pop('SalePrice').values
X = df.values

'''
Used RandomForestRegressor to get feature importances, splits on features that provide most information gain
'''
random_selector = RandomForestRegressor()
random_selector.fit(X,y)
print sorted(zip(columns,random_selector.feature_importances_),
key=lambda x: x[1])

'''
Lasso backward selection, takes away features by p-value
'''
lasso = Lasso()
rfe = RFE(model,10)
rfe = rfe.fit(X,y)
print sorted(zip(columns,rfe.support_),
key=lambda x: x[1])
print sorted(zip(columns,rfe.ranking_),
key=lambda x: x[1])

'''
Used GradientBoostingRegressor, similar concept to RF feature selection
'''
boost_selector = GradientBoostingRegressor()
boost_selector.fit(X,y)
print sorted(zip(columns,boost_selector.feature_importances_),
key=lambda x: x[1])
