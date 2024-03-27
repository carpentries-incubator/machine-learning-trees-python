---
title: "Random forest"
teaching: 20
exercises: 10
questions:
- "How can subselection of variables improve performance?"
objectives:
- "Train a random forest model."
- "Visualise the decision boundaries."
keypoints:
- "With Random Forest models, we resample data and use subsets of features."
- "Random Forest are powerful predictive models."
---

## Random Forest

In the previous example, we used bagging to randomly resample our data to generate "new" datasets. The Random Forest takes this one step further: instead of just resampling our data, we also select only a fraction of the features to include.

It turns out that this subselection tends to improve the performance of our models. The odds of an individual being very good or very bad is higher (i.e. the variance of the trees is increased), and this ends up giving us a final model with better overall performance (lower bias).

Let's train the model.

```python
np.random.seed(321)
mdl = ensemble.RandomForestClassifier(max_depth=5, n_estimators=6, max_features=1)
mdl = mdl.fit(x_train.values, y_train.values)

fig = plt.figure(figsize=[12,6])
for i, estimator in enumerate(mdl.estimators_):    
    ax = fig.add_subplot(2,3,i+1)
    txt = 'Tree {}'.format(i+1)
    glowyr.plot_model_pred_2d(estimator, x_train, y_train, title=txt)
```

![](../fig/section6-fig1.png){: width="900px"}

> ## Question
> a) When specifying the model, we set `max_features` to `1`. All of the trees make decisions using both features, so it appears that our model is not respecting the argument.  What is the explanation for this inconsistency?
> b) What would you expect to see with a `max_features` of `1` AND a `max_depth` of `1`? 
> c) Repeat the plots with the new argument to check your answer to b. What do you see with respect to Age? Why?
> > ## Answer
> > a) If it was true that setting `max_features=1` as an argument led to trees with a single variable, we would not see the trees in our figure (which all make decisions based on both features). The explanation is that features are being limited at each split, not at the model level.
> > b) Setting `max_features` to `1` limits our trees to a single split. We now see two sets of trees, some restricted to Acute Physiology Score and some restricted to Age.
> > c) Our trees decided against splitting on Age. The model was unable to find a single Age that led to improvement (based on its optimisation criteria).
> {: .solution}
{: .challenge} 

Let's look at final model's decision surface.

```python
plt.figure(figsize=[9,5])
txt = 'Random forest (final decision surface)'
glowyr.plot_model_pred_2d(mdl, x_train, y_train, title=txt)
```

![](../fig/section6-fig2.png){: width="900px"}

Again, the visualization doesn't really show us the power of Random Forests, but we'll quantitatively evaluate them soon enough.

{% include links.md %}

