# Gauusian Process with DPP-Nystrom

This repository includes the source code for Gaussian process regression combined with the Nystrom approxomation by k-DPP sampling. The results of the numerical experiments can be checked in the "./demo.ipynb".

## Demo on Gaussian Process with Nystrom Method

For the landmark selection of the Nystrom method, following 4 methods were compared;
* uniform sampling
* k-DPP (by Gibbs sampling)
* simmulated annealing of the MAP of k-DPP
* greedy algorithm for the likelihood maximization
The result can be replicated by running the jupyter notebook contained in the main directory.  The aileron datasetused here is taken from [https://sci2s.ugr.es/keel/dataset.php?cod=93](https://sci2s.ugr.es/keel/dataset.php?cod=93).
![](fig/summary.png)

---

## Prerequisites
* `Python 3.7`
* `numpy`
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
