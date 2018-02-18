# Kaggle competition
House Prices: Advanced Regression Techniques

+ [*Kaggle competition*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submissions?sortBy=date&group=all&page=1)

![House Prices: Advanced Regression Techniques](https://kaggle2.blob.core.windows.net/competitions/kaggle/5407/media/housesbanner.png)

*With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges us to predict the final price of each home*.

+ **Goal:**
Our job is to predict the sales price for each house. For each Id in the test set, we must predict the value of the SalePrice variable. 

+ **Data Tidying**:
  1. First of all we worked on Train set. We checked the distribution of SalePrice (the variable of interest) and thanks to the documentation we deleted the outliers. After that we transformed SalePrice in a log-scale.

  2. We dropped the ID column both in Train and Test because useless for predictions. We concatenated the two datasets.
  
  3. We made the imputation for the categorical variables where **Na** were occurred and then we created dummies variables. We splitted again the whole dataset in Train and Test.

+ **Feature engineering**:
We assigned the train dataset to X variable and the SalePrice column to y variable.

Computing each time an arithmetic mean between the models we obtained the following results.

+ **Model selection**:

**GB** ==> (loss = *huber*, learning_rate = .05, n_estimators = 5000, max_depth = 1) 

**Lasso** ==> (alpha = 0.000476269)

**Ridge** ==> (alpha = 10.971561867)

**LGBM** ==> (*regression*ânum_leaves = 4, learning_rate =0.03,n_estimators=30000, max_bin = 50) 

*Score: 0.11627*
