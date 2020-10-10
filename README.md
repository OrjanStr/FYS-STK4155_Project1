# FYS-STK4155_Project1 Linear regression models

## General info
Repository for the first project in FYS-STK 3155
the main code is located in code/Linear_regression.py
note that the plots in visuals may differ from those in report 

## Abstract 
We consider three types of linear regression,  ordinary least squares method (OLS), Ridge regression and Lasso regression and look at generated data with Franke's function and a terrain data set. With python, using our own implementations as well as functionality from sci-kit Learn, we minimised the cost function for these three methods. The Franke dataset consists of $n=400$ datapoints. Terrain originally had $n=6485401$ but it was reduced to $n=6486$ datapoints. The best method for franke data was Lasso as it yielded the lowest MSE. For terrain data the best method turned out to be OLS since for terrain data overfitting is not an issue, due to the large amount of data points in terrain data. We were unable to plot and analyse the methods for high complexities because of computation time. The methods we examined work well with a limited dataset, but other methods might work better on terrain data.

## How to Run Code
* install pipenv on your system 
* clone the repository
*  in correct folder, type:
```
install pipenv
```
* enter shell:
```
pipenv shell
```
* run code file as normal


## Table of contents
* [Visuals](code/visuals)
* [Code](code)
* [Example runs](code/example_runs)
* [Report](report)


## example use 
To run examples with Franke data, run the individual files. 
to run examples with terrain data, use terrainmap.py
#### task_a.py:
MSE and R2 score, as well as beta variance for OLS
#### task_b.py:
  use OLS and bootstrap to produce Training error, Test error plot
and Bias Variance plot.
#### task_c.py:
compare bootstrap with kfold for OLS
#### task_d.py:
plotting heatmap for bootstrap and kfold, and bias variance with ridge
#### task_e.py:
plot bootsrap error, kfold error bias and variance for Lasso
## test of code/benchmarks
to test the function write
```
pytest -v 
```
or run the program normally. this also serves as benchmarks to check that each part of the code is running as expected.
