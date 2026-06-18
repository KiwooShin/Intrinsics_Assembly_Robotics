"""Generate N single-trial SFP configs with small randomization NEAR the eval values.
Keeps the proven trial_1 topology (nic_rail_2 / sfp_port_0) so CheatCode reliably
succeeds; perturbs goal (board+card), grasp (end), and robot start (home joints).
"""
import sys, os, copy, random, argparse
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import yaml

SRC = '/home/kiwoos/work/Intrinsics_Assembly_Robotics/aic_engine/config/eval_config.yaml'

def perturb(base, i, seed=0, mode='near'):
    rng = random.Random(seed + i)
    U = rng.uniform
    c = copy.deepcopy(base)
    c['trials'] = {'trial_1': copy.deepcopy(base['trials']['trial_1'])}
    tb = c['trials']['trial_1']['scene']['task_board']
    task = c['trials']['trial_1']['tasks']['task_1']
    if mode == 'near':
        # --- GOAL: board pose (position + yaw), tight around eval trial_1 ---
        tb['pose']['x'] += U(-0.015, 0.015)
        tb['pose']['y'] += U(-0.015, 0.015)
        tb['pose']['yaw'] += U(-0.08, 0.08)
        # target stays nic_rail_2 / sfp_port_0
        nic = tb['nic_rail_2']['entity_pose']
        nic['translation'] = U(-0.0215, 0.0234)
        nic['yaw'] = U(-0.04, 0.04)
    else:  # wide: full SFP eval distribution
        tb['pose']['x'] = 0.16 + U(-0.03, 0.04)        # eval x ~0.15-0.20
        tb['pose']['y'] = -0.21 + U(-0.02, 0.26)       # eval y ~-0.22..0.05
        tb['pose']['yaw'] = 3.1 + U(-0.2, 0.2)         # board faces robot (~pi)
        rail = rng.choice([0, 1, 2, 3, 4])             # random target NIC rail
        for k in range(5):
            r = tb[f'nic_rail_{k}']
            if k == rail:
                r['entity_present'] = True
                r['entity_name'] = f'nic_card_{k}'
                r['entity_pose'] = {'translation': U(-0.0215, 0.0234),
                                    'roll': 0.0, 'pitch': 0.0, 'yaw': U(-0.04, 0.04)}
            else:
                tb[f'nic_rail_{k}'] = {'entity_present': False}
        task['port_name'] = rng.choice(['sfp_port_0', 'sfp_port_1'])
        task['target_module_name'] = f'nic_card_mount_{rail}'
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
    ap.add_argument('--mode', choices=['near', 'wide'], default='near')
    ap.add_argument('--prefix', default='cfg')
    a = ap.parse_args()
    os.makedirs(a.o, exist_ok=True)
    base = yaml.safe_load(open(SRC))
    print(f"mode={a.mode}  {'cfg':5} {'bx':>7} {'by':>7} {'yaw':>6} {'rail':>5} {'port':>11}")
    for i in range(a.n):
        c = perturb(base, i, a.seed, a.mode)
        p = f'{a.o}/{a.prefix}_{i}.yaml'
        yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
        tb = c['trials']['trial_1']['scene']['task_board']
        task = c['trials']['trial_1']['tasks']['task_1']
        rail = next((k for k in range(5) if tb[f'nic_rail_{k}'].get('entity_present')), '?')
        print(f"      {i:<5} {tb['pose']['x']:7.3f} {tb['pose']['y']:7.3f} {tb['pose']['yaw']:6.2f} "
              f"{str(rail):>5} {task['port_name']:>11}  -> {p}")

if __name__ == '__main__':
    main()
