### Minimum-Volume-Ellipsoid Estimator

The Minimum-Volume-Ellipsoid Estimator is an estimator used to detect outliers given that a sample is from the population with the multivariate normality.   
The Mahalanobis distance method and this returns very similar result when the sample size is large, thanks to the Strong Law of Large Numbers.   
However when the sample size is small, the Mahalanobis distanace is likely to be affected by the outliers and detects outliers incorrectly.   
**Rousseeuw and van Zomeren** in 1990 introduced the MVE estimator and as part of the class presentation, I created a python script MVE.py that takes a pandas dataframe and returns an object.
For more explanation with examples, please take a look at the `ipynb` file in the repo.
