import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import randint
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA




def print_lasso(comb_data, y):
    for dat in comb_data:
        X= dat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        alphas = np.arange(0.1, 10, 0.1)
        lasso_cv_uns = LassoCV(alphas=alphas, cv=15)
        lasso_cv_uns.fit(X_train, y_train)
        opt_alpha_lasso = lasso_cv_uns.alpha_
        print(f"for unscaled data: \n ")
        print(f"Optimal alpha: {lasso_cv_uns.alpha_}")
        print(f"Number of features used: {np.sum(lasso_cv_uns.coef_ != 0)}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_cv_uns.predict(X_test))}")
        print(f"R^2 score: {lasso_cv_uns.score(X_test, y_test)}")
        print(f"_ _ "*10)
        alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas)
        #plt.figure(figsize=(10, 10))
        #for i in range(coefs_lasso.shape[0]):
        #    plt.plot(alphas_lasso, coefs_lasso[i], label=f"Feature {i+1}")
        #plt.axvline(opt_alpha_lasso, color="black", linestyle="--", label="Optimal Alpha")
        #plt.xlabel("Alpha")
        #plt.ylabel("Coefficient")
        #plt.title("Lasso Path")
        ##plt.legend()
        #plt.show()
        lasso_cv_uns = LassoCV(alphas=alphas, cv=15)
        lasso_cv_uns.fit(X_train_scaled, y_train)
        opt_alpha_lasso1 = lasso_cv_uns.alpha_
        print(f"for scaled data: \n")
        print(f"Optimal alpha: {lasso_cv_uns.alpha_}")
        print(f"Number of features used: {np.sum(lasso_cv_uns.coef_ != 0)}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_cv_uns.predict(X_test_scaled))}")
        print(f"R^2 score: {lasso_cv_uns.score(X_test_scaled, y_test)}")
        print(f"_ _ "*10)
        print(f" _ _"*10)
        alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas)
        #plt.figure(figsize=(10, 10))
        #for i in range(coefs_lasso.shape[0]):
        #    plt.plot(alphas_lasso, coefs_lasso[i], label=f"Feature {i+1}")
        #plt.axvline(opt_alpha_lasso1, color="black", linestyle="--", label="Optimal Alpha")
        #plt.xlabel("Alpha")
        #plt.ylabel("Coefficient")
        #plt.title("Lasso Path")
        ##plt.legend()
        #plt.show()
    print(f"_________________________________________________________")
        
def print_elastic_net(comb_data, y):
    for dat in (comb_data):
        X=dat     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        alphas = np.logspace(-4, 0, 50)
        elastic_net_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=5)
        elastic_net_cv.fit(X_train, y_train)
        print(f"for unscalled data :\n ")
        print(f"Optimal alpha: {elastic_net_cv.alpha_}")
        print(f"Optimal l1_ratio: {elastic_net_cv.l1_ratio_}")
        print(f"Number of features used: {np.sum(elastic_net_cv.coef_ != 0)}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, elastic_net_cv.predict(X_test))}")
        print(f'R^2 value: {elastic_net_cv.score(X_test, y_test)}')
        print(f"_ _ "*10)
    
        elastic_net_cv = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=5)
        elastic_net_cv.fit(X_train_scaled, y_train)
        print(f"for scalled data :\n ")
        print(f"Optimal alpha: {elastic_net_cv.alpha_}")
        print(f"Optimal l1_ratio: {elastic_net_cv.l1_ratio_}")
        print(f"Number of features used: {np.sum(elastic_net_cv.coef_ != 0)}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, elastic_net_cv.predict(X_test_scaled))}")
        # also print r^2 value
        print(f'R^2 value: {elastic_net_cv.score(X_test, y_test)}')
        print(f"_ _ "*10)
        print(f" _ _"*10)
    print(f"_________________________________________________________")
        
def pritn_lasso_text(df_text,y):
    x=df_text.drop(['Item Difficulty',
                    'Item Discrimination',
                    'Item Type',
                    'InternCode',
                    'Title',
                    'Content',
                    'Question',
                    'Correct Response',
                    'Response Option 1',
                    'Response Option 2',
                    'Response Option 3',
                    'Response Option 4',
                    'Response Option 5',
                    'Response Option 6',
                    'Response Option 7',
                    'Item Type'],axis=1)
    x.replace('Missing',0,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = np.arange(0.1, 10, 0.1)
    lasso_cv_uns = LassoCV(alphas=alphas, cv=15)
    lasso_cv_uns.fit(X_train, y_train)
    opt_alpha_lasso = lasso_cv_uns.alpha_
    print(f"for unscaled data: \n ")
    print(f"Optimal alpha: {lasso_cv_uns.alpha_}")
    print(f"Number of features used: {np.sum(lasso_cv_uns.coef_ != 0)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_cv_uns.predict(X_test))}")
    print(f"R^2 score: {lasso_cv_uns.score(X_test, y_test)}")
    print(f"_ _ "*10)

    lasso_cv_uns = LassoCV(alphas=alphas, cv=15)
    lasso_cv_uns.fit(X_train_scaled, y_train)
    opt_alpha_lasso = lasso_cv_uns.alpha_
    print(f"for scaled data: \n ")
    print(f"Optimal alpha: {lasso_cv_uns.alpha_}")
    print(f"Number of features used: {np.sum(lasso_cv_uns.coef_ != 0)}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_cv_uns.predict(X_test_scaled))}")
    print(f"R^2 score: {lasso_cv_uns.score(X_test_scaled, y_test)}")
    print(f"_________________________________________________________")

def print_random_forest(comb_data, y):
    for dat in (comb_data):
        X=dat     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) # Basic Model
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f'R^2 value: {rf.r2_score(y_pred, y_test)}')
        print(f"_ _ "*10)
        # Tuned Model
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_features = ['log2', None, 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [20, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
        rf_hp = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        rf_random.fit(X_train, y_train)
        y_pred = rf_random.predict(X_test)
        print(f"Mean Squared Error of Hypertuned Model: {mean_squared_error(y_test, y_pred)}")
        print(f'R^2 value of Hypertuned Model: {rf.r2_score(y_pred, y_test)}')
        print(f"_ _ "*10)
        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        principalComponents = pca.fit_transform(X_scaled)
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        n_components_80 = (cumulative_variance >= 0.80).argmax() + 1
        pca = PCA(n_components=n_components_80)
        principalComponents = pca.fit_transform(X_scaled)
        pca_X = pd.DataFrame(
             data=principalComponents, 
             columns=[f'PC{i+1}' for i in range(n_components_80)]
             )
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(pca_X, y, test_size=0.2, random_state=42)
        rf.fit(X_train_pca, y_train)
        y_pred = rf.predict(X_test_pca)
        print(f"Mean Squared Error of PCA Model: {mean_squared_error(y_test, y_pred)}")
        print(f'R^2 value of Hypertuned Model: {rf.r2_score(y_pred, y_test)}')
        print(f"_ _ "*10)
        print(f" _ _"*10)
    print(f"_________________________________________________________")

def print_SVR(comb_data, y):
    for dat in (comb_data):
        X=dat     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        kernel_functions = ['linear', 'rbf']
        for i, kernel in enumerate(kernel_functions, 1):
            print (kernel_functions[i-1])
            svr_classifier = SVR(kernel=kernel)
            svr_classifier.fit(X_train, y_train)
            y_pred = svr_classifier.predict(X_test)
            print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
            print(f'R^2 value: {svr_classifier.r2_score(y_pred, y_test)}')
            print(f"_ _ "*10)
            # Hypertuned Model
            param_grid = {'linear': {'C': [0.1, 1, 10, 100]},'rbf': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1], 'epsilon': [0.01, 0.1, 0.5, 1]} }
            for kernel in ['linear', 'rbf']:
                print(f"Performing hyperparameter tuning for {kernel} kernel")
            svr = SVR(kernel=kernel)
            grid_search = GridSearchCV(svr, param_grid[kernel], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            print(f"Mean Squared Error of Hypertuned Model: {mean_squared_error(y_test, y_pred)}")
            print(f'R^2 value of Hypertuned Model: {svr.r2_score(y_pred, y_test)}')
            print(f"Best parameters for {kernel} kernel {grid_search.best_params_}")
            print(f"_ _ "*10)
            print(f" _ _"*10)
    print(f"_________________________________________________________")

def print_SVM(comb_data, y):
    for dat in (comb_data):
        X=dat     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
        }
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5) 
        grid.fit(X_train_scaled, y_train)
        y_pred = grid.predict(X_test_scaled)
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
        print(f"_ _ "*10)
        print(f" _ _"*10)
    print(f"_________________________________________________________")

def print_Random_Forest_Classification(comb_data, y):
    for dat in (comb_data):
        X=dat     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
        print(f"_ _ "*10)
        # Hypertuned Model
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_features = ['log2', None, 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [20, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_dist = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}
        rf = RandomForestClassifier(random_state=42)
        rand_search = RandomizedSearchCV(rf, 
                                         param_distributions = param_dist, 
                                         n_iter=5, 
                                         cv=5, 
                                         random_state=42)
        rand_search.fit(X_train, y_train)
        y_pred = rand_search.predict(X_test)
        print(f'Best hyperparameters:',  rand_search.best_params_)
        print(f"Accuracy Hypertuned Model: {metrics.accuracy_score(y_test, y_pred)}")
        print(f"_ _ "*10)
        # Feature Importance
        importances = rf.feature_importances_
        selector = SelectFromModel(rf, prefit=True, threshold="mean")
        X_train_reduced = selector.transform(X_train)
        X_test_reduced = selector.transform(X_test)
        rf_reduced = RandomForestClassifier( random_state=42)
        rf_reduced.fit(X_train_reduced, y_train)
        print(f"Accuracy on reduced feature set (Features): {rf_reduced.score(X_test_reduced, y_test)}") 
        print(f"_ _ "*10)
        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        optimal_components = (cumulative_variance >= 0.95).argmax() + 1 #95% explained variance
        print(f"Optimal number of components to retain 95% variance: {optimal_components}") 
        print(f"_ _ "*10)
        pca = PCA(n_components=optimal_components)
        principalComponents = pca.fit_transform(X_scaled)
        pca_X = pd.DataFrame(
            data=principalComponents, 
            columns=[f'PC{i+1}' for i in range(principalComponents.shape[1])]
        )
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(
            pca_X, y, test_size=0.2, random_state=42
        )
        rf.fit(X_train_pca, y_train)
        y_pred_sc = rf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred_sc)
        print(f"Accuracy after selecting optimal components:", accuracy) 
        print(f"_ _ "*10)
        print(f"_ _ "*10)
    print(f"_________________________________________________________")
    



        