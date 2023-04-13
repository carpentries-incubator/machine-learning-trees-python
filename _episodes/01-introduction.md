---
title: "Introduction"
teaching: 20
exercises: 10
questions:
- "What steps are needed to prepare data for analysis?"
- "How do I create training and test sets?"
objectives:
- "Load the patient data."
- "Explore summary characteristics of the data."
- "Prepare the data for analysis."
keypoints:
- "Understanding your data is key."
- "Data is typically partitioned into training and test sets."
- "Setting random states helps to promote reproducibility."
---

## Predicting the outcome of critical care patients

We would like to develop an algorithm that can be used to predict the outcome of patients who are admitted to intensive care units using observations available on the day of admission.

Our analysis focuses on ~1000 patients admitted to critical care units in the continental United States. Data is provided by the Philips eICU Research Institute, a critical care telehealth program.

We will use decision trees for this task. Decision trees are a family of intuitive "machine learning" algorithms that often perform well at prediction and classification.

## Load the patient cohort

We will begin by loading a set of observations from our critical care dataset. The data includes variables collected on Day 1 of the stay, along with outcomes such as length of stay and in-hospital mortality.

```python
# import libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# load the data
cohort = pd.read_csv('./eicu_cohort_trees.csv')

# Display the first 5 rows of the data
cohort.head()
```

The data has been assigned to a dataframe called `cohort`. Let's take a look at the first few lines:

|index|gender|age|admissionweight|unabridgedhosplos|acutephysiologyscore|apachescore|actualhospitalmortality|heartrate|meanbp|creatinine|temperature|respiratoryrate|wbc|admissionheight|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|Female|48|86\.4|27\.5583|44|49|ALIVE|102\.0|54\.0|1\.16|36\.9|39\.0|6\.1|177\.8|
|1|Female|59|66\.6|15\.0778|56|61|ALIVE|134\.0|172\.0|1\.03|34\.8|32\.0|25\.5|170\.2|
|2|Male|31|66\.8|2\.7326|45|45|ALIVE|138\.0|71\.0|2\.35|37\.2|34\.0|21\.4|188\.0|
|3|Female|51|77\.1|0\.1986|19|24|ALIVE|122\.0|73\.0|-1\.0|36\.8|26\.0|-1\.0|160\.0|
|4|Female|48|63\.4|1\.7285|25|30|ALIVE|130\.0|68\.0|1\.1|-1\.0|29\.0|7\.6|172\.7|

## Preparing the data for analysis

We first need to do some basic data preparation. 

```python
# Encode the categorical data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cohort['actualhospitalmortality_enc'] = encoder.fit_transform(cohort['actualhospitalmortality'])
```

In the eICU Research Database, ages over 89 years are recorded as ">89" to comply with US data privacy laws. For simplicity, we will assign an age of 91.5 years to these patients (this is the approximate average age of patients over 89 in the dataset).

```python
# Handle the deidentified ages
cohort['age'] = pd.to_numeric(cohort['age'], downcast='integer', errors='coerce')
cohort['age'] = cohort['age'].fillna(value=91.5)
```

Now let's use the tableone package to review our dataset.

```python
!pip install tableone

from tableone import tableone

t1 = tableone(cohort, groupby='actualhospitalmortality')
print(t1.tabulate(tablefmt = "github"))
```

The table below shows summary characteristics of our dataset:

|                                    |         | Missing   | Overall      | ALIVE        | EXPIRED      |
|------------------------------------|---------|-----------|--------------|--------------|--------------|
| n                                  |         |           | 536          | 488          | 48           |
| gender, n (%)                      | Female  | 0         | 305 (56.9)   | 281 (57.6)   | 24 (50.0)    |
|                                    | Male    |           | 230 (42.9)   | 207 (42.4)   | 23 (47.9)    |
|                                    | Unknown |           | 1 (0.2)      |              | 1 (2.1)      |
| age, mean (SD)                     |         | 0         | 63.4 (17.4)  | 62.2 (17.4)  | 75.2 (12.6)  |
| admissionweight, mean (SD)         |         | 16        | 81.8 (25.0)  | 82.3 (25.1)  | 77.0 (23.3)  |
| unabridgedhosplos, mean (SD)       |         | 0         | 5.6 (6.8)    | 5.7 (6.7)    | 4.3 (7.8)    |
| acutephysiologyscore, mean (SD)    |         | 0         | 41.7 (22.7)  | 38.5 (18.8)  | 74.3 (31.7)  |
| apachescore, mean (SD)             |         | 0         | 53.6 (25.1)  | 49.9 (21.1)  | 91.8 (30.5)  |
| heartrate, mean (SD)               |         | 0         | 101.5 (32.9) | 100.3 (31.9) | 113.9 (40.0) |
| meanbp, mean (SD)                  |         | 0         | 89.6 (41.5)  | 90.7 (40.7)  | 78.8 (47.6)  |
| creatinine, mean (SD)              |         | 0         | 0.8 (2.0)    | 0.8 (2.0)    | 1.4 (1.8)    |
| temperature, mean (SD)             |         | 0         | 35.6 (5.6)   | 35.9 (4.8)   | 32.9 (10.4)  |
| respiratoryrate, mean (SD)         |         | 0         | 27.4 (15.5)  | 26.8 (15.4)  | 33.9 (15.2)  |
| wbc, mean (SD)                     |         | 0         | 6.5 (7.6)    | 6.2 (7.1)    | 9.9 (11.2)   |
| admissionheight, mean (SD)         |         | 8         | 168.4 (14.5) | 168.2 (13.6) | 170.3 (21.5) |
| actualhospitalmortality_enc, n (%) | 0       | 0         | 488 (91.0)   | 488 (100.0)  |              |
|                                    | 1       |           | 48 (9.0)     |              | 48 (100.0)   |

> ## Question
> a) What proportion of patients survived their hospital stay?  
> b) What is the "apachescore" variable?  Hint, see the [Wikipeda entry for the Apache Score](https://en.wikipedia.org/wiki/APACHE_II ).  
> c) What is the average age of patients?   
> > ## Answer
> > a) 91% of patients survived their stay. There is 9% in-hospital mortality.   
> > b) APACHE ("Acute Physiology and Chronic Health Evaluation II") is a severity-of-disease classification system. It is applied within 24 hours of admission of a patient to an intensive care unit. Higher scores correspond to more severe disease and a higher risk of death.    
> > c) The median age is 64 years. Remember that the age of patients above 89 years is unknown. Median is therefore a better measure of central tendency. The median age can be calculated with `cohort['age'].median()`.
> {: .solution}
{: .challenge} 

## Creating train and test sets

We will only focus on two variables for our analysis, age and acute physiology score. Limiting ourselves to two variables (or "features") will make it easier to visualize our models. 

```python
from sklearn.model_selection import train_test_split

features = ['age','acutephysiologyscore']
outcome = 'actualhospitalmortality_enc'

x = cohort[features]
y = cohort[outcome]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state =  42)
```

> ## Question
> a) Why did we split our data into training and test sets?   
> b) What is the effect of setting a random state in the splotting algorithm?    
> > ## Answer
> > a) We want to be able to evaluate our model on data that it has not seen before. If we evaluate our model on data that it is trained upon, we will overestimate the performance.    
> > b) Setting the random state means that the split will be deterministic (i.e. we will all see the same "random" split). This helps to ensure our analysis is reproducible.   
> {: .solution}
{: .challenge} 

{% include links.md %}

