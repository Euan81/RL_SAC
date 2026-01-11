Please use gait_ref_6d_mujoco_hipRneg.npz
  (arrays under keys hip_R,knee_R,ankle_R,hip_L,knee_L,ankle_L in radians, already in MuJoCo Walker2d joint sign conventions).

gait_ref_6d.npz --> measured joint angles
gait_ref_6d_mujoco.npz --> transformed to mujoco format, although the hip sign needed to be flipped (hence the recommended use of "gait_ref_6d_mujoco_hipRneg.npz")
