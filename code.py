import pandas as pd
import collections as col
import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf
import random as rnd
import itertools as itr
import xgboost as xgb

# Data - preprocessing
tr = pd.read_csv("train_1.csv")
page_split = np.array([x.rsplit('_', 3) for x in tr.iloc[:,0].values])
tr["name"] = page_split[:,0]
tr["project"] = page_split[:,1]
tr["access"] = page_split[:,2]
tr["agent"] = page_split[:,3]

tr.iloc[:,1:-4].to_csv("train_1_mod.csv", sep=',', index=True)
tr.iloc[:,[0, -4, -3, -2, -1]].to_csv("map_1.csv", sep=',', index=True)

tr = pd.read_csv("map_1.csv")
tr.columns = ['id', 'page', 'name', 'project', 'access', 'agent']
tr.to_csv('map_1.csv', sep=',', index=False)

tr = pd.read_csv("train_1_mod.csv")
cols = ['id']
cols.extend(list(tr.columns[1:]))
tr.columns = cols
tr.to_csv('train_1_mod.csv', sep=',', index=False)
del cols

# Data - insights
tr_s = pd.read_csv("train_1_mod.csv")
null_counts = tr_s.isnull().sum(axis=1).values
plt.hist(null_counts[null_counts!=0])
plt.xlabel('Null Counts')
plt.ylabel('# pages')
plt.title('Distribution of Null Counts')
plt.show()
print("# No missing data : " + str((null_counts==0).sum()) + " : " + str(float((null_counts==0).sum())/len(null_counts)))

tr_m = pd.read_csv("map_1.csv")
col.Counter(tr_m.project)
col.Counter(tr_m.access)
col.Counter(tr_m.agent)
col.Counter([tuple(x) for x in tr_m.iloc[:,[3,4,5]].values])

# Data Modelling
tr_s = pd.read_csv("train_1_mod.csv")
tr_s = tr_s[tr_s.isnull().sum(axis=1) == 0]

for i in range(len(tr_s)):
    data = tr_s.iloc[[i], 1:].T
    data.columns = ['count']
    plt.plot(data.iloc[:,0].values)
    plt.show()
    
    best_models = {}
    
    # ARIMA
    print('--- ARIMA ---')
    ars = range(100)
    mas = range(10)
    ins = range(5)
    combs = list(itr.product(ars, mas, ins))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.ARIMA(data = data, ar=comb[0], ma=comb[1], integ=comb[2], target='count', family=pf.Normal())
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['ARIMA'].aic if 'ARIMA' in best_models else np.inf):
            print(str(comb))
            best_models['ARIMA'] = mdl
    
    print(best_models['ARIMA'].summary())
    print('\n')
    # DAR
    print('--- DAR ---')
    ars = range(100)
    ins = range(5)
    combs = list(itr.product(ars, ins))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.DAR(data = data, ar=comb[0], integ=comb[1], target='count')
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['DAR'].aic if 'DAR' in best_models else np.inf):
            print(str(comb))
            best_models['DAR'] = mdl

    print(best_models['DAR'].summary())
    print('\n')
    # GARCH
    print('--- GARCH ---')
    data_GARCH = pd.DataFrame(np.diff(np.log(data['count'].values)))
    data_GARCH.index = data.index.values[1:data.index.values.shape[0]]
    data_GARCH.columns = ['count']
    ps = range(10)
    qs = range(10)
    combs = list(itr.product(ps, qs))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.GARCH(data = data, p=comb[0], q=comb[1], target='count')
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['GARCH'].aic if 'GARCH' in best_models else np.inf):
            print(str(comb))
            best_models['GARCH'] = mdl
    
    print(best_models['GARCH'].summary(transformed=False))
    print('\n')
    # GAS
    print('--- GAS ---')
    ars = range(10)
    scs = range(5)
    ins = range(5)
    combs = list(itr.product(ars, scs, ins))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.GAS(data = data, ar=comb[0], sc=comb[1], integ=comb[2], target='count', family=pf.Normal())
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['GAS'].aic if 'GAS' in best_models else np.inf):
            print(str(comb))
            best_models['GAS'] = mdl
    
    print(best_models['GAS'].summary())
    print('\n')
    # GAUSS_L
    print('--- GAUSS_L ---')
    ins = range(5)
    combs = list(itr.product(ins))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.LocalLevel(data = data, integ=comb[0], target='count', family=pf.Normal())
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['GAUSS_L'].aic if 'GAS' in best_models else np.inf):
            print(str(comb))
            best_models['GAUSS_L'] = mdl
    
    print(best_models['GAUSS_L'].summary())
    print('\n')
    # GAUSS_LT
    print('--- GAUSS_LT ---')
    ins = range(5)
    combs = list(itr.product(ins))
    combs = rnd.sample(combs, int(0.05*len(combs)))
    for comb in combs:
        mdl = pf.LocalTrend(data = data, integ=comb[0], target='count', family=pf.Normal())
        mdl = mdl.fit('MLE')
        print(str(comb) + ' : AIC - ' + str(mdl.aic) + ' | BIC - ' + str(mdl.bic))
        
        if(np.isnan(mdl.aic)):
            continue
        
        if(mdl.aic < best_models['GAUSS_LT'].aic if 'GAS' in best_models else np.inf):
            print(str(comb))
            best_models['GAUSS_LT'] = mdl
    
    print(best_models['GAUSS_LT'].summary())
    print('\n')
    
    # Data preprocessing for NLM
    # pv - # Past Values to be used to predict Future values
    # fv - # Future Values to predict
    pv = 30
    fv = 1
    raw_data = data.iloc[:,0].values
    proc_data = np.array([raw_data[i:i+pv+fv] for i in range(raw_data.shape[0]-pv-fv+1)])
    X = proc_data[:, :pv]
    y = proc_data[:, pv:]
    if fv == 1:
        y = [:, 0]
        
    # Non-Linear Modelling
    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)
    model = GridSearchCV(refit=False, estimator=rf, param_grid=dict(n_estimators=[10, 100], max_features=[X.shape[1], int(sqrt(X.shape[1]))], bootstrap=[True, False], random_state=[0, 1, 2]), n_jobs=-1)
    model = model.fit(X, y)
    print("Best Score : "+str(model.best_score_))
    print("Bset Params : "+str(model.best_params_))
    print("Best Estimator : "+str(model.best_estimator_))
    print("\n")
    
    # SVR
    svr = SVR(kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    model = GridSearchCV(refit=False, estimator=svr, param_grid=dict(C=[0.1, 0.3, 1, 3, 10], epsilon=[0.3, 1, 3, 10, 30]), n_jobs=-1)
    model = model.fit(X, y)
    print("Best Score : "+str(model.best_score_))
    print("Bset Params : "+str(model.best_params_))
    print("Best Estimator : "+str(model.best_estimator_))
    print("\n")
    
    #
