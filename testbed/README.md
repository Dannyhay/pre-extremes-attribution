# Testbed experiments (Secs. II D, III, IV B, VI; Figs. 1, 2, 4, 5, 6; Tables I and II)

All experiments use the soil-moisture--temperature testbed of Sec. VI A. Conditional
on a soil path the temperature is Gaussian, so densities, currents, exceedance
probabilities and adjoints are evaluated exactly from a Gaussian mixture over soil
paths (`common.py`); only the operating-envelope experiments, whose worlds break
that structure, use Monte Carlo over the temperature noise.

| Script | Reproduces | Output |
|---|---|---|
| `stationary_regimes.py` | Liang flow and bridge identity in the wet, transitional and very dry regimes; stationary currents; independent-source check (Secs. II D, VI A-B; Fig. 1; Fig. 2(a),(b)) | `stationary.json` |
| `traversal.py` | Exceedance-flux identity along a wet-to-dry traversal; cumulative budget; non-attribution of the integrated terms; ground-truth counterfactual (Secs. III, VI B-C; Fig. 2(c); Fig. 4) | `traversal.json` |
| `adjoint_accuracy.py` | Adjoint against the exact counterfactual: absolute vs relative reading, epsilon sweep, six independent ensembles, analytic standard error (Secs. IV B, V C, VI D; Fig. 5) | `adjoint.json` |
| `envelope.py` | Fitted-generator attribution: control, inert correlated decoy, sample size, hidden slow memory, state-dependent coupling, multiplicative noise (Sec. VI E; Fig. 6; Table I) | `envelope.json` |
| `diagnostics.py` | Residual autocorrelation and portmanteau statistic in each world; fitted relaxation rate against sampling step; true source function versus tent basis with 14, 28, 56 knots (Sec. VI E) | `diagnostics.json` |
| `saturation.py` | Very dry and transitional regimes with ensemble-mean, climatological and full-removal references; first-order and exact responses; Gaussian mean-shift prediction (Sec. VI F; Table II) | `saturation.json` |
| `make_figs.py` | Figs. 1, 2, 4, 5, 6 from the JSON files | PDFs |

Run in order: `python stationary_regimes.py && python traversal.py && python adjoint_accuracy.py && python envelope.py && python diagnostics.py && python saturation.py && python make_figs.py`.
Each script fixes its random seed. Run times on a laptop: roughly 1-25 min each.

Design choices: the traversal moves the soil equilibrium linearly from logit(0.74)
to logit(0.04) between days 10 and 130; the adjoint, envelope and saturation windows
are 5 days long (comparable to the lead times of Sec. VII A) and start from a
stationary regime; thresholds are fixed values chosen from the base rates listed in
each script; the ensemble-mean perturbation moves the source forcing a fraction
epsilon = 0.25 of the way to its ensemble mean; the climatological reference of
`saturation.py` is the mean forcing of the transitional regime (1.476 degC/day).
