# FAST-PT Perturbation-Theory Tables

This repository implements the `Cobaya` theory block `fastpt`: the python
FAST-PT package (McEwen et al 2016 https://arxiv.org/abs/1603.04826,
Fang et al 2017 https://arxiv.org/abs/1609.05978) computing the
perturbation-theory power spectra that Cosmolike's TATT
intrinsic-alignment model (Blazek et al 2019
https://arxiv.org/abs/1708.09247) and one-loop galaxy-bias expansion
consume. Cosmolike computes the same integrals with cfastpt, the C code
inside the compiled interface; the likelihood key `IA_code` selects the
implementation. cfastpt (`IA_code: 0`) is the reference implementation
and the default; this block is the `IA_code: 1` path, and the projects
compare the two in a unit test (the CFASTPT vs FASTPT comparison of each
project's `tests/README.md`).

## Installation

Cocoa installs both repositories via environmental keys on
`set_installation_options.sh`: the upstream FAST-PT package at a pinned
commit and this repository (linked into Cobaya as the theory `fastpt`)
at a pinned tag. One key, `IGNORE_FASTPT_CODE`, governs both. Users must
ensure the following line is commented out in
`set_installation_options.sh` before running `setup_cocoa.sh` and
`compile_cocoa.sh`. *By default, this line should be commented out, but
it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_FASTPT_CODE=1

      (...)

      export FASTPT_URL="https://github.com/jablazek/FAST-PT.git"
      export FASTPT_GIT_COMMIT="a970d700703814f8513cf3848add4c167e60fc76"
      export FASTPT_NAME="FAST-PT"

      export FASTPT_WRAPPER_URL="https://github.com/CosmoLike/fastpt.git"
      export FASTPT_WRAPPER_NAME="PyFAST-PT"
      export FASTPT_WRAPPER_GIT_TAG="v1.0"

> [!Warning]
> Do not `pip install` the FAST-PT package directly. Cocoa pins its
> commit and seeds its Python dependencies with guarded versions; a
> direct pip install can upgrade numpy/scipy on `.local` and break the
> environment.

## Usage

Running TATT with FAST-PT requires two additions to the YAML file.

**Step :one:**: select TATT and the FAST-PT implementation on the
Cosmolike likelihood:

```yaml
likelihood:
  roman_real.cosmic_shear:
    IA_model: 1 # NLA (0) or TATT (1)
    IA_code: 1  # cfastpt (0) or FAST-PT through this block (1)
```

**Step :two:**: add the theory block:

```yaml
theory:
  fastpt:
    path: ./external_modules/code/FAST-PT
    extra_args:
      accuracyboost: 1.0
      internal_accuracyboost: 1.0
      kmax_boltzmann: 7.5
      extrap_kmax: 250.0
```

The values above are the defaults; `path` is the only required key.

| `extra_args` key | default | what it sets |
|---|---|---|
| `accuracyboost` | 1.0 | multiplies the density of the output table the likelihood reads with linear interpolation |
| `internal_accuracyboost` | 1.0 | multiplies the density of the internal grid the FFTLog convolutions run on |
| `kmax_boltzmann` | 7.5 | the $k_{\rm max}$ in $1/{\rm Mpc}$ the Boltzmann code computes $P(k)$ to; the boosts do not scale it |
| `extrap_kmax` | 250.0 | the reach in $1/{\rm Mpc}$ of the matter-power interpolator's log-extrapolation, which serves the grids beyond `kmax_boltzmann` |

The two boosts multiply two different grids. The output table is what
the likelihood interpolates linearly, so its density controls the
accuracy of every TATT and bias term; the internal grid only feeds the
convolutions, whose computed terms a cubic spline in $\log k$ upsamples
onto the output table. Both defaults are the converged configuration
(the decision record lives in `projects/lsst_y1/tests/README.md`), so
raising either boost is a convergence test, not a need.

> [!NOTE]
> Before the two-grid upgrade of this block (2026-09) one shared
> 1,100-point grid played both roles, and across the TATT prior the
> FAST-PT and cfastpt data vectors differed by up to
> $\Delta\chi^2 = 21$ at LSST-Y1 precision and $\Delta\chi^2 = 87184$
> at Roman precision. Separating the grids located the difference in
> the density of the interpolated table and none of it in the
> convolutions.

> [!Warning]
> The boost values of the old semantics (`accuracyboost: 80`, `640`,
> `5120`) counted grid points and are retired: under the rebased
> semantics they would build tables of $10^8$ points, so the block
> refuses `accuracyboost` above 8 and prints the reason. Set the boosts
> near 1.

> [!NOTE]
> roman_fourier's precision needs `accuracyboost: 2.0`, the value its
> example yamls recommend; every other project passes its comparison
> test at the defaults. Each project pins its configuration in that
> test (see the CFASTPT vs FASTPT comparison of its
> `tests/README.md`).

## Design

The block computes the intrinsic-alignment tables (tidal alignment,
tidal torquing, and their mixed terms) and the one-loop galaxy-bias
tables at $z = 0$ from the linear $P(k)$ the Boltzmann code provides.
The likelihood hands the tables to the compiled interface, which reads
them with linear interpolation in $\log k$. At the default boosts the
grids are

    output table:  about 10^6 points, log-spaced
    internal grid: 1,100 points, log-spaced, same k range

The convolutions run on the internal grid and the cubic spline
upsamples their terms onto the output table, so the dense table costs
almost nothing per sample: the internal-grid strategy the `bfmt` theory
block also uses. The linear $P(k)$ row of the output table comes
directly from the Boltzmann interpolator, never from the spline. The
Boltzmann request stays at `kmax_boltzmann`; the grids' reach beyond it
is served by the interpolator's log-extrapolation up to `extrap_kmax`,
the same regime cfastpt's own $P(k)$ input lives in.
