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

def print_lasso(comb_data, y):
    for emb in comb_data:
        X=emb
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
        plt.figure(figsize=(10, 10))
        for i in range(coefs_lasso.shape[0]):
            plt.plot(alphas_lasso, coefs_lasso[i], label=f"Feature {i+1}")
        plt.axvline(opt_alpha_lasso, color="black", linestyle="--", label="Optimal Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Coefficient")
        plt.title("Lasso Path")
        plt.legend()
        plt.show()
        lasso_cv_uns = LassoCV(alphas=alphas, cv=15)
        lasso_cv_uns.fit(X_train_scaled, y_train)
        opt_alpha_lasso1 = lasso_cv_uns.alpha_
        print(f"for scaled data: \n")
        print(f"Optimal alpha: {lasso_cv_uns.alpha_}")
        print(f"Number of features used: {np.sum(lasso_cv_uns.coef_ != 0)}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, lasso_cv_uns.predict(X_test_scaled))}")
        print(f"R^2 score: {lasso_cv_uns.score(X_test_scaled, y_test)}")
        print(f"_________________________________________________________")
        print(f"_________________________________________________________")
        alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas)
        plt.figure(figsize=(10, 10))
        for i in range(coefs_lasso.shape[0]):
            plt.plot(alphas_lasso, coefs_lasso[i], label=f"Feature {i+1}")
        plt.axvline(opt_alpha_lasso1, color="black", linestyle="--", label="Optimal Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Coefficient")
        plt.title("Lasso Path")
        plt.legend()
        plt.show()
        
def print_elastic_net(comb_data, y):
    for emb in (comb_data):
        X=emb     
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
        print(f"_________________________________________________________")
        print(f"_________________________________________________________")
        