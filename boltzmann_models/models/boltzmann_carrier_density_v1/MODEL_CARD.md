---
craft_id: boltzmann_carrier_density_v1
version: 0.1.0
project: boltzmann
license: Apache-2.0
tags: ['solution_type:DC', 'model:charge_transport', 'family:pn_diode']
library_name: boltzmann-models
---

# boltzmann_carrier_density_v1

**Project:** boltzmann · **Version:** 0.1.0 · **License:** Apache-2.0
**Produced by:** [Intel — Universal ML Inference Engine](https://huntingtonai.com)

> boltzmann inference surrogate

> ⚠️ **Synthetic / non-engine-truth data (`cited`).** The metrics below are
> **NOT** the real simulation engine's output — they are computed on the craft's synthetic
> or cited training distribution. The headline number is the **`release metric`** metric, not an
> engine-faithful accuracy; read it as such.

## Model description

`boltzmann_carrier_density_v1` is a **surrogate model** ("craft") used by the AHL Intel inference
engine to replace expensive numerical simulation with fast neural inference.
Output kind: **fields** (6 outputs). boltzmann inference surrogate

- **Status:** trained
- **Type:** neural inference surrogate (physics-trained)
- **Parameters:** 97,286
- **Weights format:** `model.safetensors` (float32)
- **Device family:** `pn_diode`

## Intended use

Drop-in inference surrogate for boltzmann. You provide a physical **device
specification** (geometry, materials, doping, operating point); the package
returns the physical output channels below. The device→model featurization and
de-normalization are handled inside the package. Not intended for regimes
outside its training domain (below).

## Inputs & outputs

**Input — physical device specification** (`input.kind: device_spec`):

| Field | What you provide |
|---|---|
| `geometry` | rectilinear grid: per-axis coords in micrometers, or shape [Nx,Ny,Nz] |
| `regions` | [{type: NA|ND, material: Si|GaAs|InGaAs, concentration: <cm^-3>}] |
| `electrodes` | optional [{id, role: anode|cathode, bounds: [[i0,i1],[j0,j1],[k0,k1]]}] |
| `operating_point` | {applied_bias_v: <V>, bias_sweep_v?: [<V>...], temperature_k?: <K>} |

The internal encoding (how the device becomes model inputs) and the normalization are handled **inside the package** — you never build feature vectors by hand.

**Output — physical fields:**

| | |
|---|---|
| **Output channels** | 6 |
| **Parameters** | 97,286 |

Output channels: `n`, `p`, `psi`, `Jx`, `Jy`, `Jz` (6 total), returned in physical units.

## Training domain / compatibility

| Property | Value |
|---|---|
| Solution type | `DC` |
| Physics models | `charge_transport` |
| Temperature | 300.0 K |
| Bias range (Vds) | 0.0–0.8 V |
| Tags | `solution_type:DC`, `model:charge_transport`, `family:pn_diode` |

## Training data

Trained on boltzmann simulation output for the `DC` solution regime (physics models: charge_transport). Labels derive from the numerical boltzmann engine / reference closed-forms; the data-generation pipeline is proprietary.

## Limitations & out-of-scope use

- **Do not** use outside the training domain above (solution type, temperature, bias range) — accuracy degrades with no guarantee.
- This is a **surrogate**, not the ground-truth solver: treat outputs as fast approximations; validate against the numerical engine where correctness is critical.
- Output kind is **fields** — consuming code must match the documented I/O contract exactly.


## Evaluation

| Metric | Value |
|---|---|
| **Held-out accuracy** | 0.851 |

_**Held-out accuracy** is an independent benchmark (`offline_eval` on a held-out set); the loss rows are the model's own recorded train/val losses. See `config.json → metrics` for the machine-readable copy._

## Caveats & recommendations

- An independent **held-out** benchmark is reported (Held-out accuracy, via `offline_eval`); the loss rows remain the model's own recorded train/val losses.
- Verify integrity before use: `shasum -a 256 -c SHA256SUMS` (or `sha256sum -c SHA256SUMS` on Linux). Weights are `safetensors` — safe to load, no pickle.
- This is a **surrogate**, not the ground-truth solver — validate against the numerical engine where correctness is critical.

## Usage

Install the inference package and run it on a physical device spec — it fetches the
weights, builds the model, and returns physical fields (no PyTorch required):

```bash
pip install boltzmann-models
```

```python
from boltzmann_models import load

model = load("boltzmann_carrier_density_v1")           # pulls + verifies weights, builds the compiled pipeline
fields = model.predict(device_spec)  # physical device in -> physical fields out
```

`device_spec` is the physical device specification documented under **Inputs & outputs**
above. Featurization and de-normalization happen inside the package.

## Provenance

- **Author:** Huntington Applied
- **Trained:** 2026-07-02T00:58:26.896922
- **Source checksum (runtime .pt):** `6790891abe30582ecec9c74133238734bae5abc3bcf66af969948b97810faed8`

## Citation

```bibtex
@software{boltzmann_carrier_density_v1_0_1_0,
  title  = {boltzmann_carrier_density_v1},
  author = {AHL / Intel},
  year   = {2026},
  note   = {Version 0.1.0},
  url    = {https://downloads.boltzmann.app/models/boltzmann_carrier_density_v1/0.1.0/}
}
```

## License

Licensed under the **Apache-2.0** (Apache-2.0) — free to use for
inference; redistribution and reverse engineering of the compiled inference core
are not permitted. See `LICENSE`.
