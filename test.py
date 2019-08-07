import pandas as pd

df = pd.read_csv('CombinedTestResultMetric.csv')
df.drop(['participantCreatedOn', 'testResultMetricCreatedOn', 'testResultMetricTimestamp1', 'testResultMetricTimestamp2', 'Unnamed: 19'], axis = 1, inplace = True)

df['testStartedAt'] = df['testStartedAt'].apply(lambda x: pd.to_datetime(x[0:-10]))
df['testEndedAt'] = df['testEndedAt'].apply(lambda x: pd.to_datetime(x[0:-10]))
df['timecomplete'] = (df['testEndedAt'] - df['testStartedAt']).apply(lambda x: x.seconds)

df['age'] = [df.testStartedAt[i].year - df.participantBirthYear[i] 
                    for i in range(0, len(df))]
df.drop(['participantBirthYear', 'testStartedAt', 'testEndedAt'], axis = 1, inplace = True)

df['participantSex'].replace(['male', 'female'], [0, 1], inplace = True)

df['participantIsControl'].replace([True, False], [1, 0], inplace = True)

mean = df.groupby(['floodlightOpenId', 'testCode', 'testMetricCode']).mean()[['testResultMetricValue', 
                                                                              'timecomplete']]
mean.rename({'testResultMetricValue': 'val_mean', 'timecomplete': 'time_mean'}, axis = 1, inplace = True)
mean_reset = mean.reset_index()
mean_reset['test'] = mean_reset['testCode'] + mean_reset['testMetricCode']
meantest = mean_reset.drop(['testCode', 'testMetricCode'], axis = 1)
meantest = meantest.set_index(['floodlightOpenId', 'test'])
meantest = meantest.reset_index()
meantest = meantest.pivot(index='floodlightOpenId', columns='test', values=['val_mean', 'time_mean'])
meantest.columns = [col[0]+col[1] for col in meantest.columns]

var = df.groupby(['floodlightOpenId', 'testCode', 'testMetricCode']).var()[['testResultMetricValue', 
                                                                              'timecomplete']]
var.rename({'testResultMetricValue': 'val_var', 'timecomplete': 'time_var'}, axis = 1, inplace = True)
var_reset = var.reset_index()
var_reset['test'] = var_reset['testCode'] + var_reset['testMetricCode']
vartest = var_reset.drop(['testCode', 'testMetricCode'], axis = 1)
vartest = vartest.set_index(['floodlightOpenId', 'test'])
vartest = vartest.reset_index()
vartest = vartest.pivot(index='floodlightOpenId', columns='test', values=['val_var', 'time_var'])
vartest.columns = [col[0]+col[1] for col in vartest.columns]

count = df.groupby(['floodlightOpenId', 'testCode', 'testMetricCode']).count()[['testResultMetricValue']]
count.rename({'testResultMetricValue': 'test_count'}, axis = 1, inplace = True)
count_reset = count.reset_index()
count_reset['test'] = count_reset['testCode'] + count_reset['testMetricCode']
counttest = count_reset.drop(['testCode', 'testMetricCode'], axis = 1)
counttest = counttest.set_index(['floodlightOpenId', 'test'])
counttest = counttest.reset_index()
counttest = counttest.pivot(index='floodlightOpenId', columns='test', values='test_count')
counttest.columns = [col+'_count' for col in counttest.columns]

dfs = [meantest, vartest, counttest]
maindf = pd.concat(dfs, axis=1)

names = ['mood', 'bottomtop', 'circle', 'figure8', 'shapehand', 'meandrawdist', 'numshapes', 'spiral', 'square',
        'topbottom', 'turnspeed', 'turns', 'ips', 'ipstime', 'bips', 'bipstime', 'mobility', 'pinchhand', 'pinches',
        'sway', 'steps']
def cols(names, metric):
    allcols = []
    for name in names: 
        allcols.append(name + '_' + metric)
    return allcols

meancols = cols(names, 'mean')
meantcols = cols(names, 'mean_t')
varcols = cols(names, 'var')
vartcols = cols(names, 'var_t')
countcols = cols(names, 'count')
allcols = meancols + meantcols + varcols + vartcols + countcols
maindf.columns = allcols

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



# tests = df.groupby(['testCode', 'testMetricCode']).size().reset_index(name="Time")
# subset = tests[['testCode', 'testMetricCode']]
# testcombs = [tuple(x) for x in subset.values]
# # This is a list of all the different tests

# def testdf(df, testcomb):
#     """
#     Takes in the complete dataframe as well as the name of each test
#     and returns a new dataframe that has all the recorded data for that test
#     """
    
#     test = df.loc[(df['testCode'] == testcomb[0]) & (df['testMetricCode'] == testcomb[1])]
#     #I am locating the rows which have data for the test that I am interested in
#     test.rename({'testResultMetricValue': '{}_val'.format(testcomb[0] + testcomb[1]), 
#                  'timecomplete':'{}_time'.format(testcomb[0] + testcomb[1])}, 
#                     axis = 1, inplace = True)
#     #I am renaming the metric value and metric time values so that it is clear which test the metrics are from
#     test.drop(['testName', 'testCode', 'testMetricName', 'testMetricCode'], 
#                   axis = 1, inplace = True)
#     #I am dropping the unnecessary columns for a cleaner dataframe
#     return test

# dailyqs_mood = testdf(df, testcombs[0])
# # Here is a sample dataframe for one of the tests