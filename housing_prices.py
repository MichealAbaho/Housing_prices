# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:50:14 2018

@author: user
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer, LabelEncoder
import scipy.stats as st
import scipy.special as st2
import matplotlib.pyplot as plt
import seaborn as sns

class housing_prices:
    
    def __init__(self, train_dset, test_dset):
        self.__train_dset = pd.read_csv(train_dset)
        self.__test_dset = pd.read_csv(test_dset)
    
    def data_prep(self):
        train = self.__train_dset
        train.set_index('Id', inplace=True)
       
        #missing values
        miss_per_col = train.isnull().sum().sort_values(ascending=False)
        miss_per_col[miss_per_col > 0].plot.bar()
        percent_miss = miss_per_col/train.isnull().count().sort_values(ascending=False)
        missing = pd.concat([miss_per_col, percent_miss], keys=['Total_missing', 'Percent'], axis=1)
        
        #Check the distribution of the SalePrice
        sns.distplot(train.SalePrice, kde=True)
        
        #check the number of purchases recorded per year
        plt.bar(train.YrSold.value_counts().index, train.YrSold.value_counts(), label='Purchases per year')
        plt.title('Purchases per year')
        plt.xlabel('Year Sold')
        plt.ylabel('# of houses')
        plt.show()
        
        #Check which House prices per year
        yr_price = train[['YrSold','SalePrice']]
        yr_price = yr_price.groupby(['YrSold'], as_index=False)['SalePrice'].sum()
        new_vals = [i for i in range(max(yr_price.YrSold)+1, (max(yr_price.YrSold) + 2))]
        newf = pd.DataFrame(new_vals, columns=['YrSold'])
        yr_pri = pd.concat([yr_price, newf])
        yr_pri.set_index('YrSold', inplace=True)
        yr_pri.fillna(0, inplace=True)
        rolling = pd.rolling_mean(yr_pri, window=2)
        yr_pri.plot()
        rolling.plot()
        plt.title('Price per annum')
        plt.xlabel('Year_sold')
        plt.ylabel('SalePrice in mlns')
        plt.show()
        
        #check the relationship between LotArea and SalePrice
        plt.scatter(train.LotArea, train.SalePrice)
        plt.title('LotArea Vs SalePrice')
        plt.show()
        
        #check the relationship between TotalBsmtSF and SalePrice
        plt.scatter(train.TotalBsmtSF, train.SalePrice)
        plt.title('Total Bsmt SF Vs SalePrice')
        plt.show()
        
        #check the relationship between GrLivArea and SalePrice
        plt.scatter(train.GrLivArea, train.SalePrice)
        plt.title('GrLivArea Vs SalePrice')
        plt.show()
        
        #check the relationship between YrBuilt(Categroical feature) and SalePrice
        yr_built = train[['SalePrice', 'YearBuilt']]
        fig, ax = plt.subplots(figsize=(25,8))
        fig=sns.boxplot(x='YearBuilt', y='SalePrice', data=yr_built)
        fig.axis(ymin=0, ymax=800000)
        plt.xticks(rotation=90)
        plt.show()
        
        #overall qualit
        ovr_qual = train[['OverallQual','SalePrice']]
        fig,ax=plt.subplots(figsize=(8,8))
        sns.boxplot(x='OverallQual', y='SalePrice', data=ovr_qual)
        plt.show()
        
        #checkout the correlation of the data 
        train_set_corr = train.corr()
        fig, ax = plt.subplots(figsize=(12,9))
        sns.heatmap(train_set_corr, linecolor='blue', square=True)
        plt.show()
        #most highly correlated variables to SalePrice
        k = 10
        cols = train_set_corr.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(train[cols].values.T)
        sns.heatmap(cm, square=True, cbar=True, annot=True,fmt='.2f', xticklabels=cols.values, yticklabels=cols.values)
        plt.show()
        
        #check normality of dependent variable
        sns.distplot(train.SalePrice, kde=True, fit=st.norm)
        fig = plt.figure()
        fig = st.probplot(train.SalePrice, plot=plt)
        
        #check normality of independent variables
        sns.distplot(train.GrLivArea, kde=True, fit=st.norm)
        fig = plt.figure()
        fig = st.probplot(train.GrLivAreae, plot=plt)
        
        sns.distplot(train.TotalBsmtSF, kde=True, fit=st.norm)
        fig = plt.figure()
        fig = st.probplot(train.TotalBsmtSF, plot=plt)
        return missing
        
        
    def study_data(self):
        #study your dataset
        self.__train_dset.describe()
        #check how many missing values exist per column
        self.__train_dset.isnull().sum()
        self.__train_dset[self.__train_dset.isnull().any(axis=1)]
        #check percentage of missing values
        percentage_missing_values = self.__train_dset.isnull().sum().sum()/np.product(self.__train_dset.shape)
        print('%d'%round(percentage_missing_values * 100)+'%')
        #visualize missing values
        self.__train_dset.isnull().sum()[self.__train_dset.isnull().sum() > 0].plot.bar()
        #visulaize the distribution of the data
        sns.distplot(self.__train_dset['SalePrice'], kde=False, fit=st.norm)
        #check some insights within the data

        train.YrSold.value_counts().plot.bar()
        #check which numeric variables have a normal distribution
        quantitative_vars = [col for col in self.__train_dset.columns if self.__train_dset.dtypes[col] != 'object']
                
        normal = pd.DataFrame(self.__train_dset[quantitative_vars])
        normal.drop(['Id', 'SalePrice'], axis=1, inplace=True) #Your not interested in normality of Id's and the Target or dependent variable
        test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01
        normal = normal.apply(test_normality)
        print(normal)
        f = pd.melt(self.__train_dset, value_vars=quantitative_vars)
        g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
        g.map(sns.distplot, 'value')
        
        
        #check which categorical variables are normally distributed using box plots
        qualitative_vars = [col for col in self.__train_dset.columns if self.__train_dset.dtypes[col] == 'object']

        #read the various groups/vcategories as type category
        for c in qualitative_vars:
            self.__train_dset[c] = self.__train_dset[c].astype('category')
            if self.__train_dset[c].isnull().any():
                self.__train_dset[c] = self.__train_dset[c].cat.add_categories(['MISSING'])
                self.__train_dset[c].fillna('MISSING')
        
        f = pd.melt(self.__train_dset, id_vars = self.__train_dset['SalePrice'], value_vars=qualitative_vars)
        g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
        g.map(sns.boxplot, 'value', 'SalePrice')
        
        def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x=plt.xticks(rotation=90)
            
        return quantitative_vars
    
    
        #quantitative_vars = [var for var in self.__train_dset if self.__train_dset.dtypes[var] != object and var != 'Id']
    
    def clean_data(self):
        train, test = self.__train_dset, self.__test_dset
        print(train.shape, test.shape)
        
        #tranforming the SalePrice Variable to achieve normality
        train['SalePrice'] = np.log(train['SalePrice'])
        
        #concatenate the two datasets
        data = pd.concat([train, test])
        data.set_index('Id', inplace=True)
# =============================================================================
#         mis_per_col = train.isnull().sum().sort_values(ascending=False)
#         percent = (train.isnull().sum()/train.isnull().count())*100
#         missing = pd.concat([mis_per_col, percent], keys=['missing', 'percent'], axis=1)
#         #dropping missing values 
#         cols_to_drop = missing[missing['missing'] > 1].index
#         train.drop(list(cols_to_drop), axis=1, inplace=True)
#         train.drop(train.loc[train['Electrical'].isnull()].index, inplace=True)
#         
#         #transforming TotalBsmtSF   
#         train['hasbmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
#         train['hasbmt'] = 0
#         train.loc[train['TotalBsmtSF'] > 0, 'hasbmt'] = 1
#         train.loc[train['hasbmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#         train.drop(['hasbmt'], axis=1)
#         #transforming the categroical vars through label encoding and fixing the skewness in the numerical variables using boxcox
#         qual_vars = [col for col in train.columns if train.dtypes[col] == object]
#         l_encoder = LabelEncoder()
#         
#         quan_vars = [col for col in train.columns if col not in qual_vars]
#         quan_vars_transform = [col for col in quan_vars if st.skew(train[col]) > 0.75]
#         
#         for cat_col in qual_vars:
#             train[cat_col] = l_encoder.fit_transform(list(train[cat_col].values))
#         
#         train[quan_vars_transform] = np.log1p(train[quan_vars_transform])
# =============================================================================
# =============================================================================
#         for num_col in quan_vars_transform:
#             train[num_col] = st.boxcox(train[num_col])
# =============================================================================
                
            
x = housing_prices('H:/Mykel/Github/Housing_prices/train.csv', 'H:/Mykel/Github/Housing_prices/test.csv')
x.clean_data()