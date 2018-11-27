# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:50:14 2018

@author: user
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import Imputer, LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
import scipy.stats as st
import scipy.special as st2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#import xgboost as xgb

#Surpress warnings from python pandas library
for i in [FutureWarning, DeprecationWarning]:
    warnings.simplefilter(action='ignore', category=i)

class housing_prices:
    
    def __init__(self, train_dset, test_dset):
        self.__train_dset = pd.read_csv(train_dset)
        self.__test_dset = pd.read_csv(test_dset)
    #
    # def data_prep(self):
    #     train = self.__train_dset
    #     train.set_index('Id', inplace=True)
    #
    #     #missing values
    #     miss_per_col = train.isnull().sum().sort_values(ascending=False)
    #     miss_per_col[miss_per_col > 0].plot.bar()
    #     percent_miss = miss_per_col/train.isnull().count().sort_values(ascending=False)
    #     missing = pd.concat([miss_per_col, percent_miss], keys=['Total_missing', 'Percent'], axis=1)
    #
    #     #Check the distribution of the SalePrice
    #     sns.distplot(train.SalePrice, kde=True)
    #
    #     #check the number of purchases recorded per year
    #     plt.bar(train.YrSold.value_counts().index, train.YrSold.value_counts(), label='Purchases per year')
    #     plt.title('Purchases per year')
    #     plt.xlabel('Year Sold')
    #     plt.ylabel('# of houses')
    #     plt.show()
    #
    #     #Check which House prices per year
    #     yr_price = train[['YrSold','SalePrice']]
    #     yr_price = yr_price.groupby(['YrSold'], as_index=False)['SalePrice'].sum()
    #     new_vals = [i for i in range(max(yr_price.YrSold)+1, (max(yr_price.YrSold) + 2))]
    #     newf = pd.DataFrame(new_vals, columns=['YrSold'])
    #     yr_pri = pd.concat([yr_price, newf])
    #     yr_pri.set_index('YrSold', inplace=True)
    #     yr_pri.fillna(0, inplace=True)
    #     rolling = pd.rolling_mean(yr_pri, window=2)
    #     yr_pri.plot()
    #     rolling.plot()
    #     plt.title('Price per annum')
    #     plt.xlabel('Year_sold')
    #     plt.ylabel('SalePrice in mlns')
    #     plt.show()
    #
    #     #check the relationship between LotArea and SalePrice
    #     plt.scatter(train.LotArea, train.SalePrice)
    #     plt.title('LotArea Vs SalePrice')
    #     plt.show()
    #
    #     #check the relationship between TotalBsmtSF and SalePrice
    #     plt.scatter(train.TotalBsmtSF, train.SalePrice)
    #     plt.title('Total Bsmt SF Vs SalePrice')
    #     plt.show()
    #
    #     #check the relationship between GrLivArea and SalePrice
    #     plt.scatter(train.GrLivArea, train.SalePrice)
    #     plt.title('GrLivArea Vs SalePrice')
    #     plt.show()
    #
    #     #check the relationship between YrBuilt(Categroical feature) and SalePrice
    #     yr_built = train[['SalePrice', 'YearBuilt']]
    #     fig, ax = plt.subplots(figsize=(25,8))
    #     fig=sns.boxplot(x='YearBuilt', y='SalePrice', data=yr_built)
    #     fig.axis(ymin=0, ymax=800000)
    #     plt.xticks(rotation=90)
    #     plt.show()
    #
    #     #overall qualit
    #     ovr_qual = train[['OverallQual','SalePrice']]
    #     fig,ax=plt.subplots(figsize=(8,8))
    #     sns.boxplot(x='OverallQual', y='SalePrice', data=ovr_qual)
    #     plt.show()
    #
    #     #checkout the correlation of the data
    #     train_set_corr = train.corr()
    #     fig, ax = plt.subplots(figsize=(12,9))
    #     sns.heatmap(train_set_corr, linecolor='blue', square=True)
    #     plt.show()
    #     #most highly correlated variables to SalePrice
    #     k = 10
    #     cols = train_set_corr.nlargest(k, 'SalePrice')['SalePrice'].index
    #     cm = np.corrcoef(train[cols].values.T)
    #     sns.heatmap(cm, square=True, cbar=True, annot=True,fmt='.2f', xticklabels=cols.values, yticklabels=cols.values)
    #     plt.show()
    #
    #
    #
    #     #check normality of dependent variable
    #     sns.distplot(train.SalePrice, kde=True, fit=st.norm)
    #     fig = plt.figure()
    #     fig = st.probplot(train.SalePrice, plot=plt)
    #
    #     #check normality of independent variables
    #     sns.distplot(train.GrLivArea, kde=True, fit=st.norm)
    #     fig = plt.figure()
    #     fig = st.probplot(train.GrLivArea, plot=plt)
    #
    #     sns.distplot(train.TotalBsmtSF, kde=True, fit=st.norm)
    #     fig = plt.figure()
    #     fig = st.probplot(train.TotalBsmtSF, plot=plt)
    #
    #
    #     return missing
    #
    #
    # def study_data(self):
    #     #study your dataset
    #     self.__train_dset.describe()
    #     #check how many missing values exist per column
    #     self.__train_dset.isnull().sum()
    #     self.__train_dset[self.__train_dset.isnull().any(axis=1)]
    #     #check percentage of missing values
    #     percentage_missing_values = self.__train_dset.isnull().sum().sum()/np.product(self.__train_dset.shape)
    #     print('%d'%round(percentage_missing_values * 100)+'%')
    #     #visualize missing values
    #     self.__train_dset.isnull().sum()[self.__train_dset.isnull().sum() > 0].plot.bar()
    #     #visulaize the distribution of the data
    #     sns.distplot(self.__train_dset['SalePrice'], kde=False, fit=st.norm)
    #     #check some insights within the data
    #
    #     train.YrSold.value_counts().plot.bar()
    #     #check which numeric variables have a normal distribution
    #     quantitative_vars = [col for col in self.__train_dset.columns if self.__train_dset.dtypes[col] != 'object']
    #
    #     normal = pd.DataFrame(self.__train_dset[quantitative_vars])
    #     normal.drop(['Id', 'SalePrice'], axis=1, inplace=True) #Your not interested in normality of Id's and the Target or dependent variable
    #     test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01
    #     normal = normal.apply(test_normality)
    #     print(normal)
    #     f = pd.melt(self.__train_dset, value_vars=quantitative_vars)
    #     g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    #     g.map(sns.distplot, 'value')
    #
    #
    #     #check which categorical variables are normally distributed using box plots
    #     qualitative_vars = [col for col in self.__train_dset.columns if self.__train_dset.dtypes[col] == 'object']
    #
    #     #read the various groups/vcategories as type category
    #     for c in qualitative_vars:
    #         self.__train_dset[c] = self.__train_dset[c].astype('category')
    #         if self.__train_dset[c].isnull().any():
    #             self.__train_dset[c] = self.__train_dset[c].cat.add_categories(['MISSING'])
    #             self.__train_dset[c].fillna('MISSING')
    #
    #     f = pd.melt(self.__train_dset, id_vars = self.__train_dset['SalePrice'], value_vars=qualitative_vars)
    #     g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    #     g.map(sns.boxplot, 'value', 'SalePrice')
    #
    #     #update numerical variables
    #     num_vars = [var for var in train_set.columns if train_set[var].dtypes != object]
    #     normality_dict = {}
    #     alpha = 0.01
    #     for var in num_vars:
    #         if(var != 'Id'):
    #             stat, p_value = sp.shapiro(train_set[var])
    #             if(p_value < alpha):
    #                 normality_dict[var] = True
    #             else:
    #                 normality_dict[var] = False
    #     normality_table = pd.DataFrame([(key, round(value,5)) for key, value in normality_dict.items()], columns=['Factor', 'Normal-Distribution'])
    #     normality_table
    #
    #     def boxplot(x,y,**kwargs):
    #         sns.boxplot(x=x,y=y)
    #         x=plt.xticks(rotation=90)
    #
    #     return quantitative_vars
    #
    #
    #     #quantitative_vars = [var for var in self.__train_dset if self.__train_dset.dtypes[var] != object and var != 'Id']
    
    def clean_data(self):
        train_set, test_set = self.__train_dset, self.__test_dset
        n_train = train_set.shape[0]
        n_test = test_set.shape[0]
        
        #tranforming the SalePrice Variable to achieve normality
        train_set['SalePrice'] = np.log(train_set['SalePrice'])
        fig = plt.figure()
        res = st.probplot(train_set['SalePrice'], plot=plt)
        plt.show()

        #concatenate the two datasets
        data = pd.concat([train_set, test_set]).reset_index(drop=True)

        #dealing with missing values, dropping all variables missing over 50% of their data
        missing_largest = (train_set.isnull().sum()/train_set.count())*100
        cols_to_drop = list(missing_largest[missing_largest>50].index)
        data.drop(cols_to_drop, axis=1, inplace=True)
        print('The data set is left with {} columns out of the original {}'.format(data.shape[1], train_set.shape[1]))

        #Now let's try and fill in missing values of those
        for i in ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','GarageCars','GarageArea','GarageYrBlt','BsmtHalfBath']:
            data[i].fillna(0, inplace=True)
        #data.drop(['Utilities'], axis=1, inplace=True)
        for i in ['MSZoning','Exterior1st','Exterior2nd','MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','GarageCond','SaleType',
                 'BsmtFinType2','Electrical','KitchenQual', 'GarageType','GarageFinish','Functional','GarageQual']:
            data[i].fillna('None', inplace=True)
        #Utilities contains only one value i.e. AllPub, which essentially isn't adding that much information to us
        data.drop(['Utilities'], axis=1, inplace=True)

        #transforming the categroical vars through label encoding
        cat_cols = data.dtypes[data.dtypes == object].index
        data[list(cat_cols)] = data[list(cat_cols)].transform(lambda x: encod(x))

        #normalize the numerical attributes
        num_cols = data.dtypes[data.dtypes == int].index

        data[list(num_cols)] = data[list(num_cols)].apply(lambda x: normalizing(x))
        clean_data = pd.get_dummies(data)
        train_data, test_data = clean_data[:n_train], clean_data[n_train:]

        return (train_data, train_set['SalePrice'], test_data)
    
    def modelling(self):
        #cross validation
        y_values = self.clean_data()[1].values
        X_train, X_test, Y_train, Y_test = train_test_split(self.clean_data()[0].values,
                                                            y_values,
                                                            test_size=0.2)
        mods = {'random-forest model':RandomForestRegressor(),
                'gradient-boosting model':GradientBoostingRegressor(learning_rate=0.1)}
        
        final_models_params = []

        for model_label, model in mods.items():
            un_tuned_batch = un_tuned_models(model, X_train, Y_train, X_test)
            train_error = un_tuned_batch[1].mean()
            test_error = mean_squared_error(Y_test, un_tuned_batch[0], multioutput='uniform_average')
            print(model_label)
            print('mean squared error during prediction: {}'.format(test_error) + '\n')

            tuned_batch = tuned_models(model, X_train, Y_train, X_test)
            train_error = tuned_batch[1].mean()
            test_error = mean_squared_error(Y_test, tuned_batch[0], multioutput='uniform_average')
            print(model_label + 'Tuned-parameters')
            print('mean squared error during training: {}'.format(train_error))
            print('mean squared error during prediction: {}'.format(test_error))
            print('Best parameters: {}'.format(tuned_batch[2]) + '\n')

            final_models_params.append(tuned_batch[2])

        return final_models_params

    def model_selection(self):



def un_tuned_models(model, x_train, y_train, xtest):
    train_model = model.fit(x_train, y_train)
    y_pred = train_model.predict(xtest)
    rmse = (-cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=5))
    return (y_pred, rmse)
        

def tuned_models(model, x_train, y_train, xtest):
    param={'n_estimators':[100, 500, 1500], 'max_depth':[None, 10, 15, 20]}
    gs = GridSearchCV(model, param, cv=5, n_jobs=-1)
    gs_model = gs.fit(x_train, y_train)
    y_pred = gs_model.predict(xtest)
    rmse = (-cross_val_score(gs_model, x_train, y_train, scoring='neg_mean_squared_error',cv=5))
    return (y_pred, rmse, gs_model.best_params_)

def encod(elem):
    lb = LabelEncoder()
    encod = lb.fit_transform(elem)
    return encod

#normalize variables
def normalizing(frame):
    normalised_dset = np.log1p(frame)
    return normalised_dset
        
#chicken()
if __name__ == '__main__':
    x = housing_prices('train.csv', 'test.csv')
    x.modelling()