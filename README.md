# Product measures under invariant determinantal processes on groups

Yitzchak Shmalo

DOI: [10.5281/zenodo.24462303](https://doi.org/10.5281/zenodo.24462303) (all versions); version 1.0: [10.5281/zenodo.24462304](https://doi.org/10.5281/zenodo.24462304)

Let Gamma be a countable group and let mu_K be the determinantal measure on {0,1}^Gamma given by a Gamma-equivariant
positive contraction K on l^2(Gamma). Lyons conjectured that mu_K stochastically dominates the Bernoulli product measure
of density equal to the Fuglede-Kadison determinant FK(K), is dominated by the product measure of density
1 - FK(I-K), and that these bounds are optimal. We show that the density FK(K) can be exceeded by a dominated product
measure for some K if and only if Gamma is nonamenable, and likewise for the upper bound. We also prove both
inequalities for every left-orderable group, and more generally for groups with a subgroup H such that Gamma/H carries
a Gamma-invariant order and H satisfies the lower inequality.

- `main.tex`, `main.pdf`: the paper.
- `checks/lemcond_check.py`, `checks/lemcond_all.py`: numerical checks of the conditioning lemma (Lemma 2.1) on a set
  of six points, by enumeration of all configurations. They need Python 3 and NumPy.

See `CREDITS.md` for the work this paper builds on.
