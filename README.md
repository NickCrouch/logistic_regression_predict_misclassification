# Logistic Regression for Identifying Potentially Misclassified Records

## Overview

This repository contains code and documentation for using logistic regression to identify potentially misclassified records in a dataset.
This approach is used to tackle the question: how do we predict whether a provider is located in Los Angeles County?
For this, the model uses the proportion of the recipient addresses that are located in LA County, and assumes that a higher proportion of recipients living in LA County denotes a provider located there also.

By fitting a logistic regression model and comparing predicted classifications to the observed labels, the workflow highlights entries that may be incorrectly labeled or warrant further review.

The repository includes code to:

* Train a logistic regression model
* Evaluate model performance
* Estimate an optimal classification threshold
* Apply the model to the full dataset
* Identify and visualize potentially misclassified records

## Visualizing Misclassifications

The plot contains:

* blue points – correctly classified observations
* red points – potential misclassifications
* logistic curve – estimated probability of service provision
* classification threshold – probability cutoff used for classification
* PROP cutoff – the implied PROP (proportion of addresses in LA County) value where the classification boundary occurs

<img width="2700" height="1800" alt="results_figure" src="https://github.com/user-attachments/assets/d693b3cf-4f07-40b2-b9ed-7e821367aa6d" />

Based on these results we can see providers where we have reason to doubt the current classification, with these being able to be reviewed further. 
