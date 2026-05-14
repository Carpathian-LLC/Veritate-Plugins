# Plugins

Plugins are how you train, refine, tune, and distill AI models in Veritate. Each one is a small folder with a script that does the work and a `manifest.json` that tells the dashboard about it. Drop a plugin in here and it shows up in the Training tab next time you open it.

## How a plugin is put together

A plugin lives in its own folder under `plugins/`. The two files that matter are:

- **`plugin.py`**. The script that runs when you click *train*. It can do whatever you want: train a model from scratch, fine-tune an existing one, run an eval, generate samples.
- **`manifest.json`**. A short file with the plugin's name, what it does, and the default values for its settings. The dashboard reads this to build the form.

Anything else in the folder is yours to use. Model code, helpers, configs, whatever. Two folder names have special meaning: `corpus/` is for training data, and `common/` is where helpers shared by more than one plugin live.

## What goes in the manifest

The manifest is just preset values that autofill in the dashboard when you train.

```json
{
  "name": "My Plugin",
  "description": "What this plugin does in one sentence.",
  "kind": "trainer",
  "flow": "scratch",
  "defaults": {
    "size": "200m",
    "precision": "bf16",
    "batch_size": 8
  }
}
```

`kind` says what your plugin does. Valid values: `"trainer"` (train a model), `"finetune"` (refine one), `"distill"` (distill one), `"eval"` (run a rubric).

`flow` is how it does it. Valid values: `"scratch"` (from zero), `"continue"` (pick up an existing checkpoint), `"adapter"` (train just an adapter on a frozen base). `flow` may also be a list (e.g. `["scratch", "continue"]`) when the same trainer supports more than one entry point.

The keys under `defaults` line up with fields the dashboard already knows how to render. Pick the ones you have an opinion about and skip the rest.

### Reserved trainer flags

A handful of `defaults` (and `args`) keys are *reserved*. The dashboard recognizes them by name and renders them with extra affordances (special label, color, tooltip, validation). Use the reserved name when your trainer supports the underlying behavior; do not invent a near-synonym. New reserved flags are added here as the platform grows. If you support a reserved flag, declare it in `defaults` (or `args`) so it shows up in the form.

| key | type | what it means | dashboard treatment |
|---|---|---|---|
| `qat_enabled` | bool | If true, the trainer wraps matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops (per-tensor maxabs INT8 weights, scale-32 INT8 activations, scale-64 LN weights) using a straight-through estimator on backprop. Result: a checkpoint whose v9 export lands with `act_boost=1` and runs cleanly on the C engine. Works on both `scratch` and `continue` flows; on `continue` it is the canonical way to repair a non-QAT checkpoint without retraining. | Rendered as a checkbox labeled **INT8 QAT** highlighted in blue. Tooltip explains the effect. Recorded into `config.json` as `"training": "qat"` so the Generation tab's Veritate warning suppresses for the resulting model. |

When a reserved flag is set, write the matching value into `args` so `save.save` records it in `config.json`. For `qat_enabled = true` the convention is `args["training"] = "qat"`.

If you need a flag the dashboard should treat as featured but it is not in the table above, file it as a contract update against `documentation/plugins/contract.md` rather than shipping a one-off. The point of the reservation is that one trainer's "INT8 QAT" toggle and another's are the *same* toggle on the dashboard.

## Hooking into the dump system (REQUIRED for trainers)

This is the contract that makes your training run show up in the dashboard. The Training tab, the loss chart, the brain panels, the lens, the candidates, the concept atlas. All of them read from the same fixed-layout files on disk. If your plugin does not write those files, the dashboard sees nothing for your model.

**The whole surface you need is two functions in one module.**

```python
from veritate_core.plugin import save, paths
```

`veritate.plugin` is the only thing a plugin is allowed to import from outside its own folder. The full contract is in [`documentation/plugins/contract.md`](../documentation/plugins/contract.md). The two calls below are the ones every trainer makes.

### Per step: `save.append_train_row`

Call this at every logging step. It appends one row to `models/<name>/train.csv`, which is what the loss chart reads.

```python
from veritate_core.plugin import save

save.append_train_row(
    name,                       # model dir name
    step,                       # int
    "train",                    # or "val"
    float(loss.item()),         # required
    lr=lr,                      # optional
    grad_norm=float(gn),        # optional
    tok_per_s=tok_per_s,        # optional
    wall_s=time.time() - t0,    # optional
    seed=args.seed,             # optional
)
```

Cheap. Call it for both train and val rows. The dashboard's loss curve, throughput chart, learning-rate chart, and gradient-norm chart all read from this one CSV.

### Per checkpoint: `save.save`

Call this every time you want to save a checkpoint. It writes the `.pt` file AND runs the full dump suite. One call, eight artifacts on disk.

```python
from veritate_core.plugin import save

ckpt_path = save.save(
    model,                      # torch.nn.Module with a vanilla state_dict()
    name,                       # model dir name
    step,                       # int
    optimizer=opt,              # optional, embedded in the .pt
    args=vars(args),            # dict; must include "description" if config.json doesn't yet exist
)
```

What lands on disk after one `save.save()` call:

```
models/<name>/
  checkpoints/
    step_<N>.pt                  the torch checkpoint
  hooks/
    step_<N>/
      probe.json                 top-K firing neurons per layer on the canonical prompt
      lens.npz                   logit-lens projections per layer
      classroom.json             param count, INT8/INT4 byte budget, weight-delta L2, alive neurons
      grades.json                reading-grade rubric scores
      concepts.json              top concept neurons per layer
      surprise.json              per-byte surprise on the canonical prompt
      quant_kl.json              KL between fp32 logits and a quantised projection
      generation.json            full TFRM v7 frame stream for the canonical prompt
```

Field schemas for every artifact and every TFRM v7 frame field live in [`documentation/hooks/contract.md`](../documentation/hooks/contract.md). That file is the contract. If you add or rename a field, you update that file in the same commit.

The eight artifacts are not optional individually. They drive different dashboard panels and the dashboard expects all eight at every saved step. If a particular dump fails (out of memory, missing corpus, bad shape) it gets logged and skipped, the checkpoint still lands. Skipping is per-failure, not per-design.

You can pass `dump_set={"surprise", "quant_kl"}` to skip specific dumps deliberately when you have a reason. Skipping by default is not the move; the dashboard will be missing the panels those dumps drive.

### What `save.save` requires from your model

- `model` is a `torch.nn.Module` whose `state_dict()` returns vanilla, unwrapped weights for a Veritate base. The dump suite assumes vanilla Veritate shapes when it builds the probe and lens passes.
- If you train a wrapper around a Veritate base (an adapter, a holographic head, a side network), call `save.save(model.base, ...)` so the dumps see the standard model. Save the wrapper state to a sidecar `.pt` next to the standard checkpoint. `multimind_m3/plugin.py::save_checkpoint` shows this pattern.
- `args` must include a non-empty `description` the first time a model is saved. After `config.json` is bootstrapped the description is sticky.

### Other helpers in `save`

| call | what it does |
|---|---|
| `save.compose_name(corpus, size, precision, version)` | builds the canonical model name `<corpus_leaf>_<size>_<precision>_<version>` |
| `save.hash_corpus(stem)` | sha256 of the corpus train (and val if present) `.bin` files; record in `config.json` to fingerprint the data |
| `save.require_description(desc)` | trims and validates a description string; raises if empty |
| `save.resolve_corpus(stem)` | returns `(train_path, val_path)` for a corpus stem; searches shared then bundled |

### The `paths` namespace

`paths` is a pure read-only helper. It builds on-disk paths so plugins do not assemble them by hand. If the platform reorganizes the layout, every plugin that uses `paths.*` keeps working; plugins that built strings themselves break.

| call | returns |
|---|---|
| `paths.model_dir(name)` | `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | path to one of the eight dump files |
| `paths.corpus_dir()` | `plugins/corpus/` |
| `paths.corpus_train_path(stem)` | `plugins/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `plugins/corpus/<stem>_val.bin` |

### What plugins must not do

- Do not import from `veritate_mri.*` directly. Use `veritate.plugin.save` and `veritate.plugin.paths`.
- Do not write outside `models/<name>/`. The dashboard reads from a fixed layout; writing elsewhere is invisible to it.
- Do not edit `config.json` after `save.save` has bootstrapped it, except via fields the contract defines.
- Do not invent your own dump artifacts. The dashboard only renders the eight in `HOOK_ARTIFACTS`. If you need a new field, add it through the [hooks contract](../documentation/hooks/contract.md) update process.

## Making a new plugin

Easiest way: copy `example_plugin/`, rename the folder, and start editing.

1. Edit the manifest. Set the name and description, pick a `kind` and `flow`, fill in any defaults you care about.
2. Write `plugin.py`. Take the CLI args the dashboard passes in, run your training loop, and at every log step call `save.append_train_row(...)`, at every checkpoint call `save.save(...)`. The example plugin shows the minimum pattern; `multimind_m3/plugin.py` shows a real run.
3. Open the dashboard and refresh the Plugins panel. Your plugin should appear in the trainer dropdown.

If something is missing (the plugin does not show up, the form looks empty, training will not start) most of the time it is that `manifest.json` is in the wrong place or has a typo. The dashboard logs will tell you.

## Training data

Your plugin reads training data from `.bin` files. Two places to put them:

- **Shared:** `plugins/corpus/<name>_train.bin` (and optionally `<name>_val.bin`). Anything here is visible to every plugin.
- **Bundled:** `plugins/<your_plugin>/corpus/<name>_train.bin`. Stays with the plugin, only that plugin sees it.

Use `save.resolve_corpus(stem)` to locate the files. It returns `(train_path, val_path)` and searches the shared folder first, then the calling plugin's bundled folder. Raises `FileNotFoundError` if no train file exists; `val_path` is `None` when there is no val file.

Build scripts that produce these files live in `plugins/common/` (e.g. `_build_pg19.py`). The `corpus/` folder itself only ever holds the `.bin` files.

## What's already here

- **`multimind_m3/`**. Byte-level transformer with the M3 holographic memory adapter. Real working trainer; copy from this for a non-trivial example of `save.save` (with sidecar) and `append_train_row`.
- **`example_plugin/`**. Minimal working trainer that demonstrates the contract. A small linear-byte model that calls `append_train_row` per step and `save.save` per checkpoint. Copy this when starting something new.

## Updating

This folder is its own git repo. To pull the latest plugins, hit *Sync* in the dashboard's Settings tab.
