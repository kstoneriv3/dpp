# Gauusian Process with DPP-Nystrom

This repository includes the source code for Gaussian process regression combined with the Nystrom approxomation by k-DPP sampling. The results of the numerical experiments can be checked in the `./demo.ipynb` or in this [report](DPP_GP.pdf). 

## Demo on Gaussian Process with Nystrom Method

The following four landmark selection scheme (of Nystrom method) were compared;
* uniform sampling
* greedy algorithm for the likelihood maximization
* k-DPP (by Gibbs sampling)
* simmulated annealing of the MAP of k-DPP

The result can be replicated by running the jupyter notebook contained in the main directory.  The dataset used here (aileron  dataset) is taken from [https://sci2s.ugr.es/keel/dataset.php?cod=93](https://sci2s.ugr.es/keel/dataset.php?cod=93).

![](fig/summary.png)

---

## Prerequisites
* `Python 3.7`
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `multiprocessing`
* `time`

## Folders/Files

```
.
├──sampler
│   ├─ __init__.py
│   ├─ dpp.py
│   ├─ greedy.py
│   ├─ mcdpp.py
│   ├─ quadrature.py
│   ├─ quadrature_back.py
│   ├─ sadpp.py
│   └─ utils.py
├──helper
│   ├─ __init__.py
│   └─ helper.py
├──data
│   └─ ailerons.txt
├──fig
│   └─ summary.png
├─ demo.ipynb
└─ README.md
```
