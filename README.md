# Using Data from Floodlight Open to Predict Multiple Sclerosis

## The data
[Floodlight Open](https://www.floodlightopen.com/en-US) is an open study by [Genentech](https://www.gene.com/) for healthy controls or patients with multiple sclerosis (MS). The goal is to be able to monitor a patient's progression over time using various tests from a smartphone app. These tests can include mood assessments, hand strength, balance, and general mobility. From a smartphone, Floodlight Open has been able to reference some of the key assessments done by neurologists on MS patients. 

## The goal
The aim of this project is to use the data collected from MS patients and healthy controls to see whether or not it is possible to predict multiple sclerosis from these tests administered by Floodlight Open. 

## Feature Engineering
[85%](https://www.nationalmssociety.org/What-is-MS/Types-of-MS/Relapsing-remitting-MS) of patients with MS are diagnosed with a disease called relapsing-remitting multiple sclerosis (RRMS). RRMS is characterized by periods of particularly bad symptoms (relapses) with periods of remission which results in slow progression. 
<p align="center">
  <img src="https://www.nationalmssociety.org/NationalMSSociety/media/MSNational/Charts-Graphics/MS_disease-course_RRMS.png">
</p>


A number of participants in the study completely multiple tests multiple times, so I calculated the mean for each participant per test. However, to preserve the nuances of the disease course of RRMS, I also calculated the variance for each test and included each individual test variance as a feature. 

I also calculated age and body mass index from the time, height, and weight data given. 

## Models used
Due to the inherent covariance associated with many of these tests, I ran a partial component analysis (PCA) followed by logistic regression, random forest, and XGBoost. However, none of these models were very strong. I then ran random forest with tuned hyperparameters on my cleaned and engineered data. This resulted in an accuracy rate of 85.47%. 

<p align="center">
  <img src="figs/RMcm.png">
</p>

I also plotted the feature importances of this model and found that variance in mobility was the most important feature. This makes a lot of sense when considering a RRMS patient's symptoms. 

<p align="center">
  <img src="figs/RFimportance.png">
</p>

## Future Applications and Importance
