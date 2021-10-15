---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---
Decision trees are a family of algorithms that are based around a tree-like structure of decision rules. These algorithms often perform well in tasks such as prediction and classification. This lesson explores the properties of tree models in the context of mortality prediction.

## Critical care data

The dataset that we will be using for this project is a subset of the [eICU Collaborative Research Database][eicu-crd] that has been created for demonstration purposes. 

The demo dataset is provided as an SQLite3 file, a relational database comprising approximately 25 tables. We will begin by loading relevant data from these tables into a single Pandas DataFrame with the help of an SQL query. 

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}

> ## Prerequisites
>
> You need to understand the basics of Python before tackling this lesson. The lesson sometimes references Jupyter Notebook although you can use any Python interpreter mentioned in the [Setup][lesson-setup].
{: .prereq}

### Getting Started

To get started, follow the directions on the "[Setup][lesson-setup]" page to download data and install a Python interpreter.

[eicu-crd]: https://doi.org/10.13026/C2WM1R

{% include links.md %}
