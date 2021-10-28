---
title: "Introduction"
teaching: 20
exercises: 10
questions:
- "How can I prepare the data for analysis?"
- "How do I create training and test sets?"
objectives:
- "Extract the patient data and prepare it for analysis."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

## Predicting the outcome of critical care patients

We would like to develop an algorithm that can be used to predict the outcome of patients who are admitted to intensive care units using observations available on the day of admission.

Our analysis focuses on 1091 patients admitted to critical care units in the continental United States. Data is provided by the Philips eICU Research Institute, a critical care telehealth program.

We will use decision trees for this task. Decision trees are a family of intuitive "machine learning" algorithms that often perform well at prediction and classification.

## Load the patient cohort

We will begin by extracting a set of observations from our critical care dataset. To help us visualise our models, we will include only two variables in our models: age and acute physiology score.

```python
import pandas as pd
import sqlite3

# prepare query
query = """
SELECT p.unitadmitsource, p.gender, p.age, p.unittype,
       a.actualhospitalmortality, a.acutePhysiologyScore
FROM patient p
INNER JOIN apachepatientresult a
ON p.patientunitstayid = a.patientunitstayid
WHERE a.apacheversion LIKE 'IVa'
AND LOWER(p.unitadmitsource) LIKE "%emergency%"
AND LOWER(p.unitstaytype) LIKE "admit%"
AND LOWER(p.unittype) NOT LIKE "%neuro%";
"""

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("data/eicu_v2_0_1.sqlite3")
cohort = pd.read_sql_query(query, con)
con.close()

# Verify that result of SQL query is stored in the dataframe
print(cohort.head())
```

The data has been assigned to a dataframe called `cohort`. Each item that is listed after the `SELECT` statement appears as a column in the data. Let's take a look at the first few lines.

```
        unitadmitsource  gender   age      unittype actualhospitalmortality  acutephysiologyscore
0  Emergency Department  Female    87  Med-Surg ICU                   ALIVE                    23
1  Emergency Department  Female    34  Med-Surg ICU                   ALIVE                    25
2  Emergency Department  Female    60  Med-Surg ICU                   ALIVE                    53
3  Emergency Department    Male    28  Med-Surg ICU                   ALIVE                    16
4  Emergency Department  Female  > 89  Med-Surg ICU                   ALIVE                    33
```
{: .output}

## Preparing the data for analysis

We first need to do some basic data preparation. 

```python
# Encode the categorical data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cohort['actualhospitalmortality_enc'] = encoder.fit_transform(cohort['actualhospitalmortality'])
```

In the eICU Collaborative Research Database, ages >89 years have been removed to comply with data sharing regulations. We will need to decide how to handle these ages. For simplicity, we will assign an age of 91.5 years to these patients.

```python
# Handle the deidentified ages
cohort['age'] = pd.to_numeric(cohort['age'], downcast='integer', errors='coerce')
cohort['age'] = cohort['age'].fillna(value=91.5)
```

Now let's use the tableone package to review our dataset.

```python
from tableone import tableone
columns = ['acutephysiologyscore','age','gender']
t1 = tableone(cohort, groupby='actualhospitalmortality')
print(t1.tabulate(tablefmt = "github"))
```

```
|                                    |                      | Missing   | Overall      | ALIVE       | EXPIRED     |
|------------------------------------|----------------------|-----------|--------------|-------------|-------------|
| n                                  |                      |           | 1012         | 937         | 75          |
| unitadmitsource, n (%)             | Emergency Department | 0         | 1012 (100.0) | 937 (100.0) | 75 (100.0)  |
| gender, n (%)                      |                      | 0         | 1 (0.1)      |             | 1 (1.3)     |
|                                    | Female               |           | 400 (39.5)   | 368 (39.3)  | 32 (42.7)   |
|                                    | Male                 |           | 611 (60.4)   | 569 (60.7)  | 42 (56.0)   |
| age, mean (SD)                     |                      | 0         | 60.9 (19.0)  | 59.9 (19.0) | 72.8 (14.0) |
| unittype, n (%)                    | CCU-CTICU            | 0         | 38 (3.8)     | 35 (3.7)    | 3 (4.0)     |
|                                    | CSICU                |           | 7 (0.7)      | 7 (0.7)     |             |
|                                    | CTICU                |           | 5 (0.5)      | 5 (0.5)     |             |
|                                    | Cardiac ICU          |           | 58 (5.7)     | 55 (5.9)    | 3 (4.0)     |
|                                    | MICU                 |           | 49 (4.8)     | 45 (4.8)    | 4 (5.3)     |
|                                    | Med-Surg ICU         |           | 836 (82.6)   | 773 (82.5)  | 63 (84.0)   |
|                                    | SICU                 |           | 19 (1.9)     | 17 (1.8)    | 2 (2.7)     |
| acutephysiologyscore, mean (SD)    |                      | 0         | 38.8 (22.0)  | 36.6 (19.2) | 66.5 (33.5) |
| actualhospitalmortality_enc, n (%) | 0                    | 0         | 937 (92.6)   | 937 (100.0) |             |
|                                    | 1                    |           | 75 (7.4)     |             | 75 (100.0)  |
```
{: .output}

## Creating train and test sets

We only focus on two variables for our analysis, age and acute physiology score. Limiting ourselves to two variables will make it easier to visualize our models.

```python
from sklearn.model_selection import train_test_split

features = ['age','acutephysiologyscore']
outcome = 'actualhospitalmortality_enc'

x = cohort[features]
y = cohort[outcome]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state =  42)
```

{% include links.md %}

