import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def robust_load_urdf(urdf_path, mesh_path_prefix):
    """
    Loads URDF by replacing 'package://ur_description/meshes/ur5' with actual mesh path.
    """
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_str = f.read()

    # Replace mesh path
    # The universalUR5.urdf uses "package://ur_description/meshes/ur5/..."
    # We want to point it to "C:/.../meshes/ur5/..."
    # We will simply replace "package://ur_description" with the parent directory of meshes folder
    # Assuming mesh_path_prefix points to "C:/.../meshes/ur5"
    
    # Use RELATIVE path to avoid encoding issues with Korean characters in absolute path using PyBullet
    # URDF is in /urdf/ folder. Meshes are in /meshes/ folder.
    # Relative path from urdf file to meshes/ur5 is "../meshes/ur5"
    
    # Common ros-style path in urdf
    package_prefix = "package://ur_description/meshes/ur5"
    
    # Relative path (ASCII safe)
    relative_mesh_path = "../meshes/ur5"
    
    new_urdf_str = urdf_str.replace(package_prefix, relative_mesh_path)
    
    # Create temp file
    temp_urdf_path = urdf_path.replace(".urdf", "_temp_auto_fixed.urdf")
    with open(temp_urdf_path, 'w', encoding='utf-8') as f:
        f.write(new_urdf_str)
        
    return temp_urdf_path

def get_joint_limits(robot_id, joint_indices):
    lower_limits = []
    upper_limits = []
    max_forces = []
    
    for j in joint_indices:
        info = p.getJointInfo(robot_id, j)
        # indices: 8=lower, 9=upper, 10=maxForce, 11=maxVelocity
        lower_limits.append(info[8])
        upper_limits.append(info[9])
        max_forces.append(info[10])
        
    return np.array(lower_limits), np.array(upper_limits), np.array(max_forces)

def get_ee_index(robot_id, ee_link_name):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode('utf-8')
        if link_name == ee_link_name:
            return i
    # Warning if not found, but universalUR5 defines it as link (associated with a joint?)
    # Valid links in PyBullet are usually associated with their parent joint index.
    # We'll check link name specifically.
    return -1

def reset_to_random_state(robot_id, joint_indices, lower_limits, upper_limits):
    max_attempts = 100
    for _ in range(max_attempts):
        random_positions = np.random.uniform(lower_limits, upper_limits)
        
        # Reset joints
        for i, j_idx in enumerate(joint_indices):
            p.resetJointState(robot_id, j_idx, random_positions[i])
            
        p.performCollisionDetection()
        contact_points = p.getContactPoints(robot_id, robot_id)
        
        if len(contact_points) == 0:
            return True, random_positions
            
    return False, None

def generate_trajectory(robot_id, joint_indices, ee_index, steps, time_step, control_limit, max_forces):
    
    # Storage
    # State: [q(6), dq(6), ee(3)] = 15 dim
    # Control: [u(6)] = 6 dim
    states = []
    controls = []
    
    # Random Control Input (Velocity)
    # The paper says "random inputs... are applied".
    # It implies the input 'u' might be constant for the track or changing?
    # "applied to the PID controller... to generate random motion data".
    # Typically in Koopman data gen, we apply random excitation. Changing every step or constant?
    # Given 0.002s step and 31 steps (0.06s), if we change u every step, it's white noise.
    # If we keep it constant, it's a step response.
    # Short duration suggests constant or slowly varying. Let's start with CONSTANT u for the short 31-step duration, 
    # or maybe change it once? "Random inputs... applied" usually means u_k is random.
    # Let's make u_k random at every step to maximize excitation richness for such short horizon.
    
    for _ in range(steps):
        # 1. Measure State (x_k)
        
        # Joint States
        joint_states = p.getJointStates(robot_id, joint_indices)
        q = [s[0] for s in joint_states]
        dq = [s[1] for s in joint_states]
        
        # EE State
        ee_state = p.getLinkState(robot_id, ee_index)
        ee_pos = ee_state[0] # (x,y,z)
        
        current_state = np.concatenate([q, dq, ee_pos]) # 15 dim
        
        # 2. Determine Control (u_k)
        # Random velocity within limits
        u_k = np.random.uniform(-control_limit, control_limit, size=len(joint_indices))
        
        # 3. Apply Control
        p.setJointMotorControlArray(
            robot_id, 
            joint_indices, 
            p.VELOCITY_CONTROL, 
            targetVelocities=u_k,
            forces=max_forces # Use parsed max forces
        )
        
        # 4. Step Simulation
        p.stepSimulation()
        
        # Store
        states.append(current_state)
        controls.append(u_k)
        
        # Wait? No, we run as fast as possible for data gen
        
    return np.array(states), np.array(controls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Dimensions check
    assert cfg['dims']['state'] == 15
    
    # Setup PyBullet
    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode) # Headless or GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load Robot
    try:
        urdf_path = cfg['robot']['urdf_path']
        mesh_path = cfg['robot']['mesh_path']
        fixed_urdf = robust_load_urdf(urdf_path, mesh_path)
        robot_id = p.loadURDF(fixed_urdf, useFixedBase=True)
    except Exception as e:
        print(f"Error loading robot: {e}")
        return

    # Get Revolute Joints
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(i)
            
    print(f"Found {len(joint_indices)} revolute joints.")
    assert len(joint_indices) == 6
    
    # Get Limits
    lower_limits, upper_limits, max_forces = get_joint_limits(robot_id, joint_indices)
    
    # Get EE Link
    ee_idx = get_ee_index(robot_id, cfg['robot']['ee_link_name'])
    if ee_idx == -1:
        # Fallback: assume last link if not found? No, safer to error.
        # But universalUR5 might have 'ee_link' as a fixed joint child. 
        # getJointInfo returns child link name.
        print(f"Error: EE link '{cfg['robot']['ee_link_name']}' not found.")
        # Debug prints
        for i in range(p.getNumJoints(robot_id)):
            print(f"{i}: {p.getJointInfo(robot_id, i)[12]}")
        return
        
    # Simulation Parameters
    dt = cfg['data']['time_step']
    p.setTimeStep(dt)
    
    # No forced damping/friction (User Request)
    # Just rely on URDF
    
    # Data params
    num_trajs = cfg['data']['num_train_trajs'] if args.mode == 'train' else cfg['data']['num_val_trajs']
    traj_len = cfg['data']['trajectory_length']
    ctrl_limit = cfg['data']['control_limit']
    
    # Buffers
    all_states = []
    all_controls = []
    
    print(f"Generating {num_trajs} trajectories for {args.mode} set...")
    
    valid_count = 0
    pbar = tqdm(total=num_trajs)
    
    while valid_count < num_trajs:
        # Reset
        success, _ = reset_to_random_state(robot_id, joint_indices, lower_limits, upper_limits)
        if not success:
            continue
            
        # Generate
        # Note: We record 31 points. 
        # x0, u0 -> x1 ... -> x30
        # If we take 31 steps, we get x1...x31?
        # Typically x0 is initial. We capture x_k.
        # Let's capture state BEFORE step, and control applied.
        # 31 data points means 31 (x, u) pairs? Or x is sequence of 31?
        # Paper: "Each track contains 31 data points".
        # Let's collect 31 states.
        
        states, controls = generate_trajectory(
            robot_id, joint_indices, ee_idx, traj_len, dt, ctrl_limit, max_forces
        )
        
        all_states.append(states)
        all_controls.append(controls)
        valid_count += 1
        pbar.update(1)
        
    pbar.close()
    
    # Save
    save_dir = cfg['data']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to big arrays
    data_X = np.array(all_states)   # [N, 31, 15]
    data_U = np.array(all_controls) # [N, 31, 6]
    
    filename = f"{save_dir}/{args.mode}_data.npz"
    np.savez_compressed(filename, states=data_X, controls=data_U)
    print(f"Saved {args.mode} data to {filename}")
    
    p.disconnect()

if __name__ == '__main__':
    main()
