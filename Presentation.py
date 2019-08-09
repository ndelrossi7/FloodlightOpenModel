import pandas as pd

# Reading in initial dataframe
df = pd.read_csv('CombinedTestResultMetric.csv')
df.drop(['participantCreatedOn', 'testResultMetricCreatedOn', 'testResultMetricTimestamp1', 'testResultMetricTimestamp2', 'Unnamed: 19'], axis = 1, inplace = True)

# Changing our time info to datetime so we can calculate age
df['testStartedAt'] = df['testStartedAt'].apply(lambda x: pd.to_datetime(x[0:-10]))
df['testEndedAt'] = df['testEndedAt'].apply(lambda x: pd.to_datetime(x[0:-10]))
df['timecomplete'] = (df['testEndedAt'] - df['testStartedAt']).apply(lambda x: x.seconds)
df['age'] = [df.testStartedAt[i].year - df.participantBirthYear[i] 
                    for i in range(0, len(df))]
df.drop(['participantBirthYear', 'testStartedAt', 'testEndedAt'], axis = 1, inplace = True)

# Making the diagnosis column clear  
df['participantIsControl'].replace([True, False], ['HC', 'MS'], inplace = True)

# Calculating the mean, variation, and count for each of the tests per participant for our feature engineering and a more accessible final dataframe.
def aggdf(df, agg, colname):
    """
    Takes in the dataframe, the kind of aggregation we want to perform, and the new column names for this dataframe and returns
    the aggregated and organized dataframe.
    """
    agg = df.groupby(['floodlightOpenId', 'testCode', 'testMetricCode']).agg('{}'.format(agg))[['testResultMetricValue']]
    agg.columns = ['{}'.format(colname)]
    agg_reset = agg.reset_index()
    agg_reset['test'] = agg_reset['testCode'] + agg_reset['testMetricCode']
    aggtest = agg_reset.drop(['testCode', 'testMetricCode'], axis = 1)
    aggtest = aggtest.set_index(['floodlightOpenId', 'test'])
    aggtest = aggtest.reset_index()
    aggtest = aggtest.pivot(index='floodlightOpenId', columns='test', values=['{}'.format(colname)])
    aggtest.columns = [col[0]+col[1] for col in aggtest.columns]
    return aggtest

meantest = aggdf(df, 'mean', 'val_mean')
vartest = aggdf(df, 'var', 'val_var')
count = df.groupby(['floodlightOpenId', 'testCode', 'testMetricCode']).count()[['testResultMetricValue']]
count.columns = ['test_count']
count_reset = count.reset_index()
count_reset['test'] = count_reset['testCode'] + count_reset['testMetricCode']
counttest = count_reset.drop(['testCode', 'testMetricCode'], axis = 1)
counttest = counttest.set_index(['floodlightOpenId', 'test'])
counttest = counttest.reset_index()
counttest = counttest.pivot(index='floodlightOpenId', columns='test', values='test_count')
counttest.columns = [col[0]+'_count' for col in counttest.columns]

dfs = [meantest, vartest, counttest]
maindf = pd.concat(dfs, axis=1)
names = ['mood', 'bottomtop', 'circle', 'figure8', 'shapehand', 'meandrawdist', 'numshapes', 'spiral', 'square',
        'topbottom', 'turnspeed', 'turns', 'ips', 'ipstime', 'bips', 'bipstime', 'mobility', 'pinchhand', 'pinches',
        'sway', 'steps']

def cols(names, metric):
    """
    Takes in the new column names and metric and returns a list of the new column names.
    """
    allcols = []
    for name in names: 
        allcols.append(name + '_' + metric)
    return allcols

meancols = cols(names, 'mean')
varcols = cols(names, 'var')
countcols = cols(names, 'count')
allcols = meancols + varcols + countcols
maindf.columns = allcols

# Organizing the categorical and constant (age, weight, height) data so that we can make one master dataframe
diagnosis = df.groupby(['floodlightOpenId', 'participantIsControl']).size().reset_index()[['floodlightOpenId', 'participantIsControl']]
sex = df.groupby(['floodlightOpenId', 'participantSex']).size().reset_index()[['floodlightOpenId', 'participantSex']]
country = df.groupby(['floodlightOpenId', 'participantCountryOfResidence']).size().reset_index(name='Time')[['floodlightOpenId', 'participantCountryOfResidence']]
age = df.groupby(['floodlightOpenId']).mean().reset_index()[['floodlightOpenId', 'age']]
weight = df.groupby(['floodlightOpenId']).mean().reset_index()[['floodlightOpenId', 'participantWeightLbs']]
height = df.groupby(['floodlightOpenId']).mean().reset_index()[['floodlightOpenId', 'participantHeightCms']]

maindf['sex'] = list(sex['participantSex'])
maindf['country'] = list(country['participantCountryOfResidence'])
maindf['age'] = list(age['age'])
maindf['weight'] = list(weight['participantWeightLbs'])
maindf['height'] = list(height['participantHeightCms'])
maindf['diagnosis'] = list(diagnosis['participantIsControl'])

def fillvar(df, metric):
    """
    Takes in dataframe and metric of interest and returns a variance of 0 in the instances where a participant
    only took a test once.
    """
    col = np.where(((maindf['{}_var'.format(metric)].isna()) & (maindf['{}_count'.format(metric)] == 1)), 
             0, maindf['{}_var'.format(metric)])
    return col

for name in names: 
    maindf['{}_var'.format(name)] = fillvar(maindf, name)
    
# Converting height and weight to BMI so that we have a more 
maindf['height'] = maindf['height'].apply(lambda x: x/100)
maindf['weight'] = maindf['weight'].apply(lambda x: x*0.454)
maindf['BMI'] = maindf['weight'] / (maindf['height'] **2)

# Dropping columns in rows where more than 50% of the data is null
thresh = len(maindf) * .5
maindf.dropna(thresh = thresh, axis = 1, inplace = True)
threshc = len(maindf.columns) * .5
maindf.dropna(thresh = threshc, axis = 0, inplace = True)

def fillna(df, metric):
    """
    Takes in the dataframe and metric and returns that column with nulls filled with imputed data per group (mean).
    """
    df = maindf['{}'.format(metric)].fillna(df.groupby('diagnosis')['{}'.format(metric)].transform('mean'))
    return df

for col in list(maindf.columns): 
    if maindf[col].isna().any():
        maindf['{}'.format(col)] = fillna(maindf, col)
        
# Dropping unnecessary and redundant columns        
maindf.drop(['shapehand_mean', 'pinchhand_mean', 'shapehand_count', 'pinchhand_count', 
             'weight', 'height', 'ipstime_count', 'bipstime_count', 'turnspeed_count'], axis = 1, inplace = True)

# Getting dummies for our categorical variables
sexdum = pd.get_dummies(maindf['sex'])
countrydum = pd.get_dummies(maindf['country'])
dummies = [sexdum, countrydum]
dummiesdf = pd.concat(dummies, axis = 1)
maindf = maindf.merge(pd.concat(dummies, axis = 1), left_index=True, right_index= True)

# Moving diagnosis column to the front for easier subsetting
maindf = maindf[['diagnosis'] + [col for col in maindf.columns if col != 'diagnosis']]

# Saving our clean dataframe to a CSV file for easier access
maindf.to_csv('maindf.csv')


###################

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go

maindf = pd.read_csv('maindf.csv')

# Visualizing the correlation of different metrics on each other
plt.figure(figsize = (20, 20))
sns.heatmap(maindf.iloc[:, 2:-13].corr(), cmap = 'Blues')

# Visualizing distribution of diagnosis
diag = px.histogram(maindf, x='diagnosis', title = 'Number of Participants with or without MS', 
                  labels = {'diagnosis': 'Diagnosis'})
# diag.show()

# Visualizing gender distribution
gen = px.histogram(maindf, x='sex', title = 'Number of Male and Female Participants',
                  labels = {'sex': 'Sex'}, color='sex')
# gen.show()

formap = [['USA', '605', 37.09, -95.71], 
          ['Canada', 115, 56.13, -106.35], 
          ['Italy', 89, 41.87, 12.56], 
          ['Switzerland', 41, 46.82, 8.22], 
          ['Spain', 26, 40.46, -3.75], 
          ['Denmark', 12, 56.26, 9.50], 
          ['Czech Republic', 2, 49.82, 15.47], 
          ['Great Britain', 2, 55.37, -3.44], 
          ['Australia', 1, -25.27, 133.78]] 
df = pd.DataFrame(formap, columns = ['Country', 'Num', 'lat', 'long']) 
pop = [int(x) for x in list(df.Num)]
lat = [float(x) for x in list(df.lat)]
long = [float(x) for x in list(df.long)]

# Visualizing distribution based on country
hmap = folium.Map(zoom_start=7)
max1 = float(max(pop))
hm_wide = HeatMap(list(zip(lat, long, pop)),
                   min_opacity=0.2,
                   max_val=max1,
                   radius=17, blur=15, 
                   max_zoom=1, 
                 )
price_map = hmap.add_child(hm_wide)

# Splitting my data
vals = maindf.iloc[:, 2:]
diag = maindf.iloc[:, 1:2]
X_train, X_test, y_train, y_test = train_test_split(vals, diag, test_size = 0.2, random_state = 42)

# Running a baseline logistic regression test
logregbase = LogisticRegression(random_state=42)
logregbase.fit(X_train, y_train)

y_pred = logregbase.predict(X_test)
training_preds = logregbase.predict(X_train)


logregbase.coef_

logregbase.intercept_

training_accuracy = accuracy_score(y_train, training_preds)
val_accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix for this model
confusion = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.heatmap(confusion, annot=True, ax = ax, cmap='Blues')

ax.set_xlabel('Predicted');ax.set_ylabel('True'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['HC', 'MS']); ax.yaxis.set_ticklabels(['HC', 'MS'])

# Running a feature importance graph for my logistic regression
logcoefficients = pd.DataFrame(logregbase.coef_).T.set_index(vals.columns)
logcoefficients.columns = ['coefficient']
logcoefficients['coefficient'] = logcoefficients['coefficient'].apply(lambda x: abs(x))
logcoefficients = logcoefficients.sort_values(by = 'coefficient')

n_features = len(logcoefficients)
plt.figure(figsize=(15,15))
plt.barh(range(n_features), logcoefficients['coefficient'], align='center') 
plt.yticks(np.arange(n_features), logcoefficients.index)
plt.title('Feature Importance in Logistic Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Running random forest on my data
forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
forest.fit(X_train, y_train)
for_pred = forest.predict(X_test)
print('Training score: {:.4}%'.format(forest.score(X_train, y_train)*100))
print('Testing score: {:.4}%'.format(forest.score(X_test, y_test)*100))

# Confusion matrix for random forest
fconfusion = confusion_matrix(y_test, for_pred)

ax = plt.subplot()
sns.heatmap(fconfusion, annot=True, ax = ax, cmap='Blues')

ax.set_xlabel('Predicted');ax.set_ylabel('True'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['HC', 'MS']); ax.yaxis.set_ticklabels(['HC', 'MS'])

# Random forest feature importance

forfeatim = pd.DataFrame(forest.feature_importances_).set_index(X_train.columns)
forfeatim.columns = ['imp']
forfeatim = forfeatim.sort_values(by = 'imp')

n_features = len(forfeatim)
plt.figure(figsize=(15,15))
plt.barh(range(n_features), forfeatim['imp'], align='center') 
plt.yticks(np.arange(n_features), forfeatim.index)
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')


# Grid search for random forest
param_grid = { 
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [10, 11, 12, 13, 14, 15, 20, 25],
    'criterion' :['gini', 'entropy']
}

grid_forest = GridSearchCV(estimator=forest, param_grid=param_grid, cv= 5)
grid_forest.fit(X_train, y_train)

grid_forest.best_params_

rfgrid=RandomForestClassifier(max_features='log2', n_estimators=400, max_depth=15, criterion='gini', random_state=42)
rfgrid.fit(X_train, y_train)
gridpred=rfgrid.predict(X_test)

# Confusion matrix for 
gridconfusion = confusion_matrix(y_test, gridpred)

ax = plt.subplot()
sns.heatmap(gridconfusion, annot=True, ax = ax, cmap='Blues')

ax.set_xlabel('Predicted');ax.set_ylabel('True'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['HC', 'MS']); ax.yaxis.set_ticklabels(['HC', 'MS'])

# feature importance for tuned random forest
forfeatimg = pd.DataFrame(rfgrid.feature_importances_).set_index(X_train.columns)
forfeatimg.columns = ['imp']
forfeatimg = forfeatimg.sort_values(by = 'imp')

n_features = len(forfeatimg)
plt.figure(figsize=(15,15))
plt.barh(range(n_features), forfeatimg['imp'], align='center') 
plt.yticks(np.arange(n_features), forfeatimg.index)
plt.title('Feature Importance in Random Forest with Hyperparameter Tuning')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Showing accuracy for other tests that I ran
fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'Accuracy']),
                 cells=dict(values=[['Logistic Regression with PCA', 'Random Forest with PCA', 'XGBoost with PCA', 
                                     'XGBoost'], 
                                    ['68.2%', '62%', '65.9%', '82.1%']]))
                     ])
# fig.show()