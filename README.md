# boltzmann-models

Runnable neural-surrogate inference for the **Boltzmann** multiphysics platform — a
physical scene in, physical fields out, in two lines. No PyTorch, no weight wrangling:
the model, featurization, and de-normalization are compiled inside the package and run
under `onnxruntime` on CPU.

```python
from boltzmann_models import load

model = load("boltzmann_carrier_density_v1")
fields = model.predict(field_grids)   # -> {"n","p","psi","Jx","Jy","Jz"} physical grids
```

`field_grids` is a mapping of physical field name → dense `[Nx, Ny, Nz]` array (a scalar
field) or `[Nx, Ny, Nz, C]` (a vector field). Which fields a craft consumes is in its
model card (`boltzmann_models/models/<craft>/MODEL_CARD.md`); fields you omit are
zero-filled. Outputs come back in physical units.

```python
import boltzmann_models as bm
bm.list_models()          # crafts shipped in this build
m = bm.load(bm.list_models()[0])
m.output_channels         # physical output channels
```

## Install

```
pip install boltzmann-models
```

Dependencies: `numpy`, `onnxruntime` (CPU). Nothing else — no torch.

## Scope of this release

- **Preview build.** This release ships the `boltzmann_carrier_density_v1` craft and
  targets **macOS (Apple Silicon, arm64) + CPython 3.14**. `pip` will only install it on a
  matching platform; more crafts and platforms follow.
- **These are fast surrogates, not the ground-truth solver.** Validate against the
  numerical engine where correctness is critical.

## License

Closed-source, compiled distribution. Licensed under the **AHL Model EULA** (see `LICENSE`).
Reverse engineering, weight/feature extraction, and redistribution are prohibited.
© 2025–2026 Huntington Applied.
