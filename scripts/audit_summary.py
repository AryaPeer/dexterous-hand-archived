"""Final summary of the math audit. Numbers came from:
  scripts/audit_geometry.py
  scripts/audit_feasibility.py
  scripts/audit_grasp_lift.py
"""
print("""
================================================================================
 SUMMARY OF MATHEMATICAL AUDIT
================================================================================

Numbers below all came from running the env code through MuJoCo (CPU), not LLM math.

ACTUATOR SET (all three envs use the same 22 actuators):
  0  slide_x_act   range=[-0.15, +0.15]  prismatic X
  1  slide_y_act   range=[-0.15, +0.15]  prismatic Y
  2-3  wrist roll/pitch (revolute)
  4-21 finger joints (revolute)
  *** THERE IS NO Z-PRISMATIC ACTUATOR ***
  -> the hand mount cannot move UP or DOWN. Lift can only come from
     finger curl re-positioning the object against the palm/fingertips.

--------------------------------------------------------------------------------
 BUG 1 (REORIENT): zero-ctrl settle releases the cube before episode starts
--------------------------------------------------------------------------------

Code at reorient_env.py:148-154:
    zero_ctrl = jnp.zeros(mjx_model.nu)
    mjx_data = mjx_data.replace(qpos=qpos, ctrl=zero_ctrl)
    mjx_data, _ = jax.lax.scan(_settle, mjx_data, None, length=5)

The hand actuators are POSITION-CONTROLLED. ctrl=0 drives joints to angle 0,
which for the finger flexion joints (ranges like [0, 1.22] and [-0.26, 1.57])
means FULLY OPEN. So the 5-step settle phase opens the fingers, and the cube
falls out before the policy ever acts.

Empirical measurement (audit_feasibility.py):
  cube spawned at grasp_site, settled 5 steps with zero_ctrl
  -> finger contacts after settle: 0 / 5
  -> cube z dropped from 0.447 -> 0.420 in 5 steps

When the SAME spawn position is settled with GRIP_BIAS ctrl (drives fingers
toward target angles 1.4-1.5 rad = closed), the cube stays in the hand:
  -> nearly all 49 spawn offsets tested showed HELD after 100 steps
  -> best position: cube z ends at 0.46-0.60, palm z = 0.41

CONSEQUENCE: every episode of the reorient sanity (3M) started with the cube
already off the hand. The policy can never recover because:
  - the hand has no Z motion (cannot reach down to the cube on the floor)
  - the cube is below the drop threshold (palm_z - 0.05 = 0.36) -> -100/step
This entirely explains nfc=0.13, ang_dist non-monotonic, success_steps~0.

FIX: replace zero_ctrl with a ctrl vector built from GRIP_BIAS. One-line
change in reorient_env.py. (Same fix should apply to peg_env.py's settle.)

--------------------------------------------------------------------------------
 BUG 2 (PEG): lift_target=10cm is geometrically impossible
--------------------------------------------------------------------------------

Empirical measurement (audit_feasibility.py):
  Hand palm z = 0.5734
  Peg initial z (on table) = 0.4390
  Peg z when held at grasp_site (pre-grasped spawn) = 0.4835
  Maximum lift achievable from finger flex alone = 4.4cm

Code at config.py:171:  lift_target: float = 0.1   # 10cm

Reward math (peg_reward.py:130):
  lift = min(lift_height / lift_target, 1.5) * contact_scale
At peg held at 4.4cm above table (the physical maximum):
  lift = 0.044 / 0.10 = 0.44   (only 44% of available reward)
At peg on table (lift = 0):
  lift = 0.0                   (zero reward)

The 4-gate insertion drive (align/depth/complete/insertion_drive) ALL multiply
through align_weight = sigmoid((peg_clearance - 0.02) * 150):
  at rest:           align_weight = 0.055
  at 2cm lift:       align_weight = 0.537
  at 4cm lift:       align_weight = 0.959  <- only fires here
  at 10cm lift:      align_weight = 1.000

The 10M sanity peg_height = 0.424 (BELOW initial 0.439), meaning across
training-time the peg was at or below the table surface on average. The
policy gripped (nfc=3) but never lifted -- because it cannot.

Stage gate also requires lift > 2cm to advance past stage 1:
  fingers_on_peg: n_contacts >= 2          (achievable)
  peg_lifted:     peg_z > initial + 0.02   (CONDITIONALLY achievable)
  peg_near_hole + aligned                   (depends on lift)

When p_pre_grasped=1.0 (curriculum stage 0), peg starts at z=0.483, so
lift_height starts at 0.044m and stage 2 fires naturally. As p_pre_grasped
drops to 0.2 (curriculum stage 4), 80% of episodes start with peg on table
at z=0.439, and stage 2 NEVER fires for those episodes.

CONSEQUENCE: this is consistent with the round-by-round failure pattern.
The eval reward peaked at 3M (282) and regressed to -6 at 10M because the
curriculum kept advancing into stages the hand cannot solve.

FIX OPTIONS:
  (a) Add a slide_z actuator to the hand mount in peg_scene_builder.py.
      This lets the policy pick up the peg vertically, matching real
      peg-in-hole physics. Cleanest.
  (b) Lower lift_target to 0.04m (4cm). Match what the hand can do.
      Also lower the align gate's bias (currently 0.02m) to ~0.01m.
  (c) Restructure the task: keep p_pre_grasped=1.0 always, so the peg is
      always pre-grasped at z=0.483. The task becomes "align + insert" and
      drops the lift requirement entirely.

--------------------------------------------------------------------------------
 BUG 3 (REORIENT): orientation reward never zero -> "do nothing" is profitable
--------------------------------------------------------------------------------

reorient_reward.py:84-86:
  soft_contact_scale = min(n_contacts / 1.0, 1.0)
  orientation_gate = 3/7 + (4/7) * soft_contact_scale
  orientation = exp(-tracking_k * ang_dist) * orientation_gate

At n_contacts = 0 (cube on floor, hand idle):
  orientation_gate = 3/7 = 0.43
  orientation reward at ang_dist=1.5 (~90 deg) = exp(-3) * 0.43 = 0.021
  with weight 7 -> 0.15 per step

Action penalty cap: -0.0002 * 20 actuators * 1.0 max = -0.004/step.

So the policy can earn ~0.15/step of "orientation" reward by doing nothing
while the cube sits on the floor. This is small but POSITIVE, and combined
with sub-floor action penalty (-0.004) creates a do-nothing local minimum.

When cube is at full drop:
  cube_drop = -20 * 1.0 * 5 = -100/step  (dominates)
But once cube hits the floor and stops bouncing, cube_drop ~= -100/step
and orientation ~= 0.15/step. Total = -99.85/step. The policy CAN'T avoid
this because the cube is unreachable -- so the gradient is "minimize
exploration" since any twitching action makes things slightly worse.

FIX: orientation reward should be 0 at nfc=0. Change line 85:
  orientation_gate = soft_contact_scale  (alpha=0 instead of 3/7)
And add a contact-acquisition shaping reward whose magnitude DOMINATES the
do-nothing policy's reward floor. E.g.:
  finger_contact_bonus = weights.contact_bonus * (n_contacts) * 5.0
This is more important AFTER fix #1 (cube held) because the contact bonus
shouldn't be hard to earn -- but it must outweigh do-nothing.

--------------------------------------------------------------------------------
 NOT A BUG: Grasp lift_target = 1.2cm is right-sized
--------------------------------------------------------------------------------

The grasp env has the same actuator set (no Z motion) but lift_target=1.2cm.
This is small enough to be reachable through dynamic finger/wrist actions
(squeeze + tilt to scoop the object against the palm). The trained policy
achieved eval_success_rate=0.25 at 5M, confirming reachability.

--------------------------------------------------------------------------------
 PROPOSED CHANGES (concrete, in priority order)
--------------------------------------------------------------------------------

1. REORIENT settle uses GRIP_BIAS ctrl (5 lines, reorient_env.py)
   - construct grip_ctrl by joint->actuator mapping at __init__
   - use grip_ctrl instead of zero_ctrl in _reset_single's settle
   - mirror in any CPU eval path if you re-add one later

2. PEG lift requirement matched to hand reach (3 options):
   - (recommended) add slide_z actuator [-0.15, 0.15] to peg_scene_builder
     -> new obs dim, new action dim. Realistic peg-in-hole physics.
   - or lower lift_target to 0.04m and align bias to 0.01m
   - or keep p_pre_grasped=1.0 forever (drop curriculum advancement)

3. REORIENT orientation gate alpha=0 (1 line, reorient_reward.py:85)
   - kills the do-nothing local minimum
   - reorient task becomes "must contact cube to earn anything"

After fixes:
  - run REORIENT 1M sanity (cheap, ~$1) to verify settle holds cube + nfc>>0
  - run PEG 1M sanity (~$1) to verify peg can be lifted (peg_height climbs)
  - if both pass at 1M, scale up

Total diagnostic spend before full run: ~$2-3 (instead of $80 per full run).

================================================================================
""")
