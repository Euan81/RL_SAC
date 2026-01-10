# Walker2d SAC Gait Tracking (clean pipeline)

This repo trains 3 SAC agents on Walker2d-v5:
- `baseline`: native environment reward only
- `paper6`: paper-style **bounded** tracking reward (paper equations), mixed with a small amount of native Walker2d reward
- `exp6`: exponential tracking variant (non-vanishing gradient), mixed with a small amount of native Walker2d reward

Key design points:
- 1 gait cycle = `gait_cycle_steps` (recommended 100 for comfortable cadence with dt=0.008s per step)
- Left leg uses a 50% phase shift (P/2) relative to right leg
- Default reference file is already in MuJoCo Walker2d joint sign conventions.

## Quick start

Install (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (comfortable pace):
```bash
python train_sac_walker2d.py --mode baseline --run-dir runs/sac_walker2d_baseline --gait-cycle-steps 100 --episode-steps 500
python train_sac_walker2d.py --mode paper6   --run-dir runs/sac_walker2d_paper6   --gait-cycle-steps 100 --episode-steps 500
python train_sac_walker2d.py --mode exp6     --run-dir runs/sac_walker2d_exp6     --gait-cycle-steps 100 --episode-steps 500
```

Evaluate:
```bash
python evaluate_run.py --run-dir runs/sac_walker2d_baseline --episodes 20
python evaluate_run.py --run-dir runs/sac_walker2d_paper6   --episodes 20
python evaluate_run.py --run-dir runs/sac_walker2d_exp6     --episodes 20
```

Compare + produce conference-ready plots/tables:
```bash
python compare_models.py --runs runs/sac_walker2d_baseline runs/sac_walker2d_paper6 runs/sac_walker2d_exp6 --out-dir runs/conference_compare --dpi 600
```

Make a video:
```bash
python make_video.py --run-dir runs/sac_walker2d_exp6 --format mp4
```

## Reference sign conversion
If you need to flip one or more joint signs (e.g., a dataset uses the opposite knee/hip convention):
```bash
python tools/test_reference_signs.py --ref references/gait_ref_6d_mujoco.npz
python tools/make_signed_reference.py --in-npz <in.npz> --out-npz references/gait_ref_6d_mujoco.npz --signs <6 numbers>
```
Then pass `--ref-npz ...` to training.


## Sign convention note

The default reference shipped with this project (`references/gait_ref_6d_mujoco.npz`) is already in MuJoCo Walker2d joint sign conventions.

If you need to adapt a reference with different conventions, do it explicitly using the provided tools (or by changing `joint_signs` in `TrainCfg`).
