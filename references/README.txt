Place your reference here.

- Default expected file: gait_ref_6d_mujoco.npz
  (arrays under keys hip_R,knee_R,ankle_R,hip_L,knee_L,ankle_L in radians, already in MuJoCo Walker2d joint sign conventions).

If you have a reference in another sign convention, you can bake in sign flips with:
  uv run python tools/make_signed_reference.py --in-npz <in.npz> --out-npz gait_ref_6d_mujoco.npz --signs ...
