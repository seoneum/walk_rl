#!/usr/bin/env python3
"""
Debug script to identify why the robot is flying/jumping
"""

import mujoco
import numpy as np
import time

def test_model():
    print("=" * 60)
    print("🔍 DEBUGGING QUADRUPED PHYSICS")
    print("=" * 60)
    
    # Load model
    model = mujoco.MjModel.from_xml_path("models/quadruped.xml")
    data = mujoco.MjData(model)
    
    print("\n📊 Model Statistics:")
    print(f"  • Number of bodies: {model.nbody}")
    print(f"  • Number of joints: {model.njnt}")
    print(f"  • Number of actuators: {model.nu}")
    print(f"  • Timestep: {model.opt.timestep}")
    print(f"  • Gravity: {model.opt.gravity}")
    
    # Check initial configuration
    print("\n🎯 Initial Position Analysis:")
    mujoco.mj_resetData(model, data)
    
    # Set initial height lower
    data.qpos[2] = 0.3  # Start at 30cm instead of 70cm
    
    # Set all joints to neutral position
    for i in range(7, model.nq):  # Skip free joint (first 7 values)
        data.qpos[i] = 0.0
    
    # Set specific joint positions for stable stance
    joint_names = ["hip_front_left", "knee_front_left", "hip_front_right", "knee_front_right",
                   "hip_back_left", "knee_back_left", "hip_back_right", "knee_back_right"]
    
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            qpos_addr = model.jnt_qposadr[joint_id]
            if "knee" in name:
                data.qpos[qpos_addr] = -0.5  # Bend knees
            else:
                data.qpos[qpos_addr] = 0.0   # Neutral hip
    
    mujoco.mj_forward(model, data)
    
    print(f"\n🤖 Robot State After Setup:")
    print(f"  • Root position: [{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")
    print(f"  • Root orientation: [{data.qpos[3]:.3f}, {data.qpos[4]:.3f}, {data.qpos[5]:.3f}, {data.qpos[6]:.3f}]")
    
    # Check feet positions
    print(f"\n👣 Feet Positions:")
    foot_names = ["foot_fl", "foot_fr", "foot_rl", "foot_rr"]
    for name in foot_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            pos = data.geom_xpos[geom_id]
            print(f"  • {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Check actuator forces
    print(f"\n⚙️ Actuator Analysis:")
    for i in range(model.nu):
        joint_id = model.actuator_trnid[2*i]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        gear = model.actuator_gear[i*6]  # First element of gear array
        ctrl_range = model.actuator_ctrlrange[i*2:(i+1)*2]
        print(f"  • Actuator {i} ({joint_name}): gear={gear:.1f}, range=[{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]")
    
    # Simulate a few steps with no control
    print(f"\n🎮 Simulating 100 steps with ZERO control...")
    for i in range(100):
        mujoco.mj_step(model, data)
        if i % 20 == 0:
            print(f"  Step {i}: z={data.qpos[2]:.3f}, vel_z={data.qvel[2]:.3f}")
    
    # Check if robot is falling or flying
    if data.qpos[2] > 0.5:
        print("  ⚠️ Robot is FLYING UP!")
    elif data.qpos[2] < 0.1:
        print("  ⚠️ Robot has FALLEN!")
    else:
        print("  ✅ Robot height seems stable")
    
    # Test with position control
    print(f"\n🎮 Testing with position control...")
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.3
    
    # Apply position control to maintain current position
    for i in range(100):
        # Set control to current joint positions (position servo)
        for j in range(model.nu):
            joint_id = model.actuator_trnid[2*j]
            qpos_addr = model.jnt_qposadr[joint_id]
            data.ctrl[j] = data.qpos[qpos_addr]
        
        mujoco.mj_step(model, data)
        if i % 20 == 0:
            print(f"  Step {i}: z={data.qpos[2]:.3f}, vel_z={data.qvel[2]:.3f}")
    
    # Final analysis
    print(f"\n📈 Final Analysis:")
    if abs(data.qvel[2]) > 1.0:
        print("  ❌ CRITICAL: Very high vertical velocity detected!")
        print("  Possible causes:")
        print("    1. Initial position conflicts with joint limits")
        print("    2. Actuator forces too high")
        print("    3. Joint ranges causing impossible configurations")
    elif data.qpos[2] < 0.05:
        print("  ⚠️ Robot collapsed to ground")
        print("  Possible causes:")
        print("    1. Joint ranges don't allow standing")
        print("    2. Initial configuration unstable")
    else:
        print("  ✅ Physics seems reasonable")
    
    return model, data

def check_joint_limits(model):
    print("\n" + "=" * 60)
    print("🔧 JOINT LIMIT ANALYSIS")
    print("=" * 60)
    
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            limited = model.jnt_limited[i]
            if limited:
                range_low = model.jnt_range[i*2]
                range_high = model.jnt_range[i*2 + 1]
                print(f"  • {name}: [{range_low:.2f}, {range_high:.2f}] rad")
                
                # Check for issues
                if "hip" in name.lower():
                    if range_low >= 0 or range_high <= 0:
                        print(f"    ⚠️ Hip should allow bidirectional movement!")
                if "knee" in name.lower():
                    if range_low > -0.1:
                        print(f"    ⚠️ Knee should allow bending (negative angles)!")

if __name__ == "__main__":
    try:
        model, data = test_model()
        check_joint_limits(model)
        
        print("\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)
        print("1. Lower initial height in config.yaml (try 0.15-0.20)")
        print("2. Reduce actuator gear values further (try 5-8)")
        print("3. Check if joint ranges allow a stable standing pose")
        print("4. Verify foot geometry doesn't penetrate ground")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure mujoco is installed:")
        print("  pip install mujoco")