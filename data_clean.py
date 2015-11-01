# /User/bin/python
# coding: utf-8

__author__ = 'Bob Li'
__date__ = '2015.10.24'

import csv
import numpy as np
import pandas as pd
import random
from time import time
import pdb
import useful_functions as udf

""" 
This script is for Kaggle Rossman Stores prediction competition
First explore data...
Pandas describe

Next clean data.....
0. Remove columns that are >95% missing

For categorical...
1. Encode categorial columns into dummy variables (one-hot encoder)

For numeric...
2. Impute missing values with median (replace NaNs with column median ignoring NaNs)
3. Remove columns that are near zero variance

Next...
1. Compute the univariate gini scores (absolute values)
2. Discard features with zero gini
3. Fit Random Forest into the data (train, cv, test)

"""
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')

print 'Reading training data...'
train_data = pd.read_csv('train.csv',parse_dates=['Date'], date_parser=dateparse)
print 'Finished reading training data...'

print 'Reading test data...'
test_data = pd.read_csv('test.csv',parse_dates=['Date'], date_parser=dateparse)
print 'Finished reading test data...'

### In Excel, split promotion interval into 3 colummns. Now convert abbreviated month into month ###
print 'Reading store data...'
store_data = pd.read_csv('store.csv')
print 'Finished reading test data...'

train_data=train_data.join(store_data,on='Store',rsuffix='_Store')
test_data=test_data.join(store_data,on='Store',rsuffix='_Store')

### Test has ID column that train does not ###
### Train has Customers + Sales column that test does not ###
### Thus, still seems train columns = test columns + 1 ###
### But be careful!!! ###
print 'Original train and test data size:'
print train_data.shape, test_data.shape

### Store in both train and test can be dropped ###
### Surprisingly, Customers is not in test dataset ###
train_data.drop(['Store','Customers','Store_Store'], axis=1, inplace=True)
test_data.drop(['Id','Store','Store_Store'], axis=1, inplace=True)

### Some data exploration ###
### Only describes int or float data types ###
print '\nTrain and test data size after dropping some useless columns:'
print train_data.shape, test_data.shape

# TO DO: Take date, obtain month (summer, winter, etc..) seasonality impacts
def add_sales_date(data):
	data['Sales_year']=data['Date'].dt.year
	data['Sales_month']=data['Date'].dt.month
	data['Sale_day']=data['Date'].dt.day
	data.drop(['Date'], axis=1, inplace=True)
	return data

train_data=add_sales_date(train_data)
test_data=add_sales_date(test_data)

print '\nTrain and test data size after obtaining dates:'
print train_data.shape, test_data.shape

###############################################################################################################
### Dealing with Categorical columns ###
### OPTION 1: Just drop categorical columns 
### OPTION 2: One-Hot Encode categorical with indicator columns (IMPLEMENTED) (USED)
cat_cols=['StateHoliday','StoreType','Assortment'] 

encode_cols=True
if encode_cols:
	### get_dummies_removeBase will return a dataframe with appropriate column names ###
	### For example: <old col name>_<category_name> ###
	df_train=train_data[cat_cols].astype(str)
	df_train=df_train.fillna(-1) ## fill with -1, else, there will be bug. get_dummies will consider this as a category

	df_test=test_data[cat_cols].astype(str)
	df_test=df_test.fillna(-1)

	### OPTIONS: Try different levels to keep. More than 50, 100, 200 unique values in a category ###
	### Tried 50 around 210 features
	dummies_train, dummies_test = udf.get_dummies_removeBase(df_train,df_test, level=50)
	pdb.set_trace
	### Now we have the dummy columns ###
	### OPTION 1: Run logistic regression L2 on these dummy to get probability and do model stacking (USED) ### 
	### OPTION 2: Remove the original cat_cols, and append dummy columns   ###
	### OPTION 3: Out-of-Fold Average (take random rolds in the Training data. Average different fold responses with each
	###									feature level)

	model_stacking=False
	if model_stacking:
		print 'Performing model stacking....'

		### JUST CALL SKLEARN AND RANDOMLY SPLIT DATA ####
		from sklearn.cross_validation import train_test_split
		stack_train_x=dummies_train.values
		stack_train_y=train_data.values[:,0].astype(int)
		stack_test_x=dummies_test.values

		x_train_A, x_train_B, y_train_A, y_train_B = train_test_split(stack_train_x,stack_train_y,test_size=0.50, random_state=42)

		print 'Size of A and B:'
		print x_train_A.shape, x_train_B.shape

		print 'running part A of stacking'
		pred_train_A=udf.ridge_dummy_regression(x_train_B,y_train_B,x_train_A,1/5.6234132519)
		print 'running part B of stacking'
		pred_train_B=udf.ridge_dummy_regression(x_train_A,y_train_A,x_train_B,1/5.6234132519)
		print 'running test set of stacking'
		pred_test=udf.ridge_dummy_regression(stack_train_x,stack_train_y,stack_test_x,1.0)

		### Now drop the original categorical columns and append the new predictions from Ridge LR
		train_data=train_data.drop(cat_cols,axis=1)
		test_data=test_data.drop(cat_cols,axis=1)

		train_stacked_col=pd.concat([pd.DataFrame(pred_train_A),pd.DataFrame(pred_train_B)])
		train_stacked_col.reset_index(drop=True, inplace=True) #Very important!! concat joins based on indices
		train_stacked_col.columns=list(['stacked_LR'])

		pred_test_col=pd.DataFrame(pred_test)
		pred_test_col.columns=list(['stacked_LR'])

		# train_data=pd.concat([train_data,stacked_col],axis=1)
		# test_data=pd.concat([test_data,pred_test_col],axis=1)
	else:
		train_data.drop(cat_cols,axis=1, inplace=True)
		train_data=pd.concat([train_data,dummies_train],axis=1)
		test_data.drop(cat_cols,axis=1,inplace=True)
		test_data=pd.concat([test_data,dummies_test],axis=1)
else:
	train_data.drop(cat_cols,axis=1,inplace=True)
	test_data.drop(cat_cols,axis=1, inplace=True)

print 'After treating categorical Train and test data shape:'
print train_data.shape, test_data.shape

print 'Filling NaN values in store data with 0...'
train_data=train_data.fillna(0)
test_data=test_data.fillna(0)
print 'Finished filling in'

### Drop rows in Train with open==0 ###
train_data = train_data[train_data.Open != 0]
train_data.drop(['Open'],axis=1,inplace=True)

###############################################################################################################
### Impute Numeric columns with median ###
y_train = train_data['Sales'].values.astype(float)
y_train=np.log(y_train)
y_train[np.isneginf(y_train)]=0

train_data.drop(['Sales'], axis=1, inplace=True)
x_train = train_data.values

x_test = test_data.values

# print 'Imputing missing values with median...'
# x_train=udf.imputeMedian(x_train) ### Self implemented, Imputer has a bug that always removes first column
# x_test=udf.imputeMedian(x_test)
# print 'Finished imputing missing values with median...'

print 'NAN values remaining (SHOULD BE 0)!'
print np.isnan(x_train).sum() #0
print np.isnan(x_test).sum() #0

print 'Train and test data feature space:'
print x_train.shape, x_test.shape
###############################################################################################################
### POSSIBLE OPTION: ###
### Remove binary columns with near zero variance ###
### near zero variance defined as 1 value takes more than 99.5% of the column ###
removeZeroVar=False
if removeZeroVar:
	nearZero_idx=udf.nearZeroVar(x_train, 0.95) ## get this to return a list of features instead...
	nearZeroCols=feature_names[nearZero_idx]
	print 'Zero variance volumns: ' 
	print nearZeroCols

	x_train=np.delete(x_train, nearZero_idx, axis=1)
	x_test=np.delete(x_test,nearZero_idx,axis=1)
	print 'Train and test data feature space after dropping nearZeroCols:'
	print x_train.shape, x_test.shape
	feature_names=np.delete(feature_names,nearZero_idx)

### Final Check ###
print 'Final Train and test data (should be equal) x_train no label here:'
print x_train.shape, x_test.shape

#### Save cleaned train and test into folder ###
# final_train=np.concatenate((np.reshape(y_train,(len(y_train),1)),x_train),axis=1)
# final_test=np.concatenate((np.reshape(y_test,(len(y_test),1)),x_test),axis=1)

# np.savetxt("bob_cleaned_train_dummies.csv", final_train, fmt='%.6e', delimiter=',', newline='\n')
# np.savetxt("bob_cleaned_test_dummies.csv", final_test, fmt='%.6e', delimiter=',', newline='\n')
##########################################################################################################

### Fit Random Forest ###
def RFR(x_train,y_train,x_test,udf_trees=100,udf_max_features='auto', udf_min_samples=1, do_CV=False,names=None):

	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_squared_error
	from sklearn.cross_validation import cross_val_score

	if do_CV:
		### Randomly split up training set into 80/20 split. ###
		### 80 for CV, 20 for "Test" score ###
		from sklearn.cross_validation import train_test_split
		x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(x_train,y_train,test_size=0.20, random_state=42)

		param_grid = {'max_features': [4,5,6],
						'min_samples_leaf':[50,250,1000,2500]}

		est=RandomForestRegressor(n_estimators=100,verbose=1, n_jobs=-1)
		cv_scores=list()
		test_scores=list()
		params_list=list()

		start = time()
		for mfeatures in param_grid['max_features']:
			for minSamples in param_grid['min_samples_leaf']:
				print 'Trying parameter combination with 100 trees: (MaxFeatures=%i, minSamples=%i)' % (mfeatures,minSamples)
				est.min_samples_leaf=minSamples
				est.max_features=mfeatures

				cv_score=cross_val_score(est,x_train_cv,y_train_cv,scoring='mean_squared_error',cv=5)
				cv_scores.append(np.mean(cv_score))

				### Create the labels for display purposes ###
				params_list.append((mfeatures,minSamples))

				### Perform 20% test set score ###
				est.fit(x_train_cv,y_train_cv)
				y_pred=est.predict(x_test_cv)
				test_scores.append(mean_squared_error(y_test_cv,y_pred))

		print 'Took %.2f seconds for parameter tuning.' %(time()-start)
		print 'writing CV results to file...'
		results = np.array([params_list,cv_scores,test_scores]).T ## should have 48 results...

		print 'Parameter tuning results........'
		print 'Parameters (max_features, min_samples_leaf), CV_Scores'
		for i in range(len(results)):
			print results[i]
	else:
		### Train the RFC Classifier with the optimal parameters found above ###
		### RFR only takes 'MSE', need to change it to RMSEPE as per contest rules ###
		print 'Fitting Random Forest with optimal user-defined parameters....'
		est=RandomForestRegressor(n_estimators=udf_trees, max_features=udf_max_features,min_samples_leaf=udf_min_samples,n_jobs=-1,verbose=1)
		est.fit(x_train,y_train)

		idx=np.where(x_test[:,1]==0)
		x_test=np.delete(x_test, 1, axis=1)
		y_pred=est.predict(x_test) 
		y_pred=np.exp(y_pred)
		y_pred[idx] = 0

		### Plot feature importances ###
		#plot_feature_importance(est, names)

		print 'Writing submission file....'
		with open('RFC_Submission.csv','wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Sales'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...' 

def GBM(x_train,y_train,x_test,udf_trees=100,udf_lr=0.01,udf_max_depth=5,udf_minsam=50,do_CV=False,names=None):
	### GridSearchCV for GradientBoostingClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.metrics import roc_auc_score

	if do_CV:
		param_grid = {'max_depth': [2,3,4,5],
						'min_samples_leaf':[50,250,1000,2500]}

		est=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1, verbose=1)
		cv_scores=list()
		params_list=list()

		start = time()
		for mdep in param_grid['max_depth']:
			for minSamples in param_grid['min_samples_leaf']:
				print 'Trying parameter combination: (Max_Depth=%i, minSamples=%i)' % (mdep,minSamples)
				est.min_samples_leaf=minSamples
				est.max_depth=mdep

				cv_score=udf.cross_val_score_proba(x_train,y_train,5,est)
				cv_scores.append(np.mean(cv_score))

				### Create the labels for display purposes ###
				params_list.append((mdep,minSamples))

		print 'Took %.2f seconds for parameter tuning.' %(time()-start)
		print 'writing CV results to file...'
		results = np.array([params_list,cv_scores]).T ## should have 48 results...

		print 'GBM Parameter tuning results........'
		print 'Parameters (max_depth, min_samples_in_leaf), CV_Scores'
		for i in range(len(results)):
			print results[i]
	else:
		### Train the GBM Classifier with the optimal parameters found above ###
		print 'Fitting GBM with optimal user-defined parameters....'
		est=GradientBoostingClassifier(n_estimators=udf_trees,learning_rate=udf_lr,max_depth=udf_max_depth,min_samples_leaf=7500,verbose=1)
		est.fit(x_train,y_train)

		y_pred=est.predict_proba(x_test)[:,1] ## Must predict probability!! ##

		### Plot feature importances ###
		plot_feature_importance(est, names)

		print 'Writing submission file....'
		with open('GBM_Submission.csv','wb') as testfile:
			w=csv.writer(testfile)
			w.writerow(('Id','Probability'))
			for i in range(len(y_pred)):
				w.writerow(((i+1),y_pred[i]))
		testfile.close()
		print 'File written to disk...'

RFR(x_train,y_train,x_test,udf_trees=100, udf_max_features=6, udf_min_samples=10,do_CV=False)
