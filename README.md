# Getting started

**floquet**  is a python package for performing floquet simulations on quantum systems to identify nonlinear resonances.

## Installation

For now we support only installing directly from github
```bash
pip install git+https://github.com/dkweiss31/floquet
```

Requires Python 3.10+

## Documentation

Documentation is available at [https://dkweiss.net/qontrol/](https://dkweiss.net/floquet/)

## Jump in

Please see the [demo](https://github.com/dkweiss31/floquet/blob/main/docs/examples/transmon.ipynb) for an example of how to set up a floquet simulation on a transmon. There we plot the "scar" plots which allow for "birds-eye view" of ionization. We then also plot the [Blais branch crossing results](https://arxiv.org/abs/2402.06615) to understand which states are responsible for the ionization. 

## Citation

If you found this package useful in academic work, please cite

```bibtex
@unpublished{floquet2024,
  title  = {Floquet: Identifying nonlinear resonances in quantum systems due to parametric drives},
  author = {Daniel K. Weiss},
  year   = {2024},
  url    = {https://github.com/dkweiss31/floquet}
}
```

Also please consider starring the project on [github](https://github.com/dkweiss31/floquet/)!
