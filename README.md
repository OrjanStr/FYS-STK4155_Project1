# FYS-STK4155_Project1 Linear regression models

## General info
the main code is located in Linear_regression.py

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
* [Report](report)
* [Visuals](visuals)
* [Example Use]()


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
