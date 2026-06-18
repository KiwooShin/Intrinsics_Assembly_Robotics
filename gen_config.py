"""Generate N single-trial SFP configs with small randomization NEAR the eval values.
Keeps the proven trial_1 topology (nic_rail_2 / sfp_port_0) so CheatCode reliably
succeeds; perturbs goal (board+card), grasp (end), and robot start (home joints).
"""
import sys, os, copy, random, argparse
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import yaml

SRC = '/home/kiwoos/work/Intrinsics_Assembly_Robotics/aic_engine/config/eval_config.yaml'

def perturb(base, i, seed=0):
    rng = random.Random(seed + i)
    U = rng.uniform
    c = copy.deepcopy(base)
    c['trials'] = {'trial_1': copy.deepcopy(base['trials']['trial_1'])}
    tb = c['trials']['trial_1']['scene']['task_board']
    # --- GOAL: board pose (position + yaw) ---
    tb['pose']['x'] += U(-0.015, 0.015)
    tb['pose']['y'] += U(-0.015, 0.015)
    tb['pose']['yaw'] += U(-0.08, 0.08)
    # --- GOAL: target NIC card slide + yaw along its rail (rail_2 stays the target) ---
    nic = tb['nic_rail_2']['entity_pose']
    nic['translation'] = U(-0.0215, 0.0234)   # task_board_limits.nic_rail
    nic['yaw'] = U(-0.04, 0.04)
    # --- END: grasp offset (~2mm / ~0.04rad real grasp deviation) ---
    g = c['trials']['trial_1']['scene']['cables']['cable_0']['pose']
    g['gripper_offset']['x'] += U(-0.002, 0.002)
    g['gripper_offset']['y'] += U(-0.002, 0.002)
    g['gripper_offset']['z'] += U(-0.002, 0.002)
    g['roll'] += U(-0.04, 0.04); g['pitch'] += U(-0.04, 0.04); g['yaw'] += U(-0.04, 0.04)
    # --- START: small robot home-joint perturbation ---
    for j in c['robot']['home_joint_positions']:
        c['robot']['home_joint_positions'][j] += U(-0.02, 0.02)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=5)
    ap.add_argument('-o', default=os.path.expanduser('~/data/configs'))
    ap.add_argument('--seed', type=int, default=20260618)
    a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True)
    base = yaml.safe_load(open(SRC))
    print(f"{'cfg':5} {'board x':>8} {'board y':>8} {'yaw':>7} {'nic_t':>7} {'nic_yaw':>8} {'grip_z':>8}")
    for i in range(a.n):
        c = perturb(base, i, a.seed)
        p = f'{a.o}/cfg_{i}.yaml'
        yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
        tb = c['trials']['trial_1']['scene']['task_board']
        g = c['trials']['trial_1']['scene']['cables']['cable_0']['pose']['gripper_offset']
        print(f"{i:<5} {tb['pose']['x']:8.4f} {tb['pose']['y']:8.4f} {tb['pose']['yaw']:7.3f} "
              f"{tb['nic_rail_2']['entity_pose']['translation']:7.4f} "
              f"{tb['nic_rail_2']['entity_pose']['yaw']:8.4f} {g['z']:8.4f}  -> {p}")

if __name__ == '__main__':
    main()
