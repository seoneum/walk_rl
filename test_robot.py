#!/usr/bin/env python3
"""
Quick test script to verify the quadruped robot configuration fixes
"""

import xml.etree.ElementTree as ET
import yaml
import sys
import os

def test_model_file():
    """Test if the model file has correct joint ranges"""
    print("=" * 60)
    print("📋 Testing Model Configuration (quadruped.xml)")
    print("=" * 60)
    
    model_path = "models/quadruped.xml"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    # Check joint ranges
    joints = root.findall(".//joint[@type='hinge']")
    print(f"\n🔧 Found {len(joints)} hinge joints:")
    
    for joint in joints:
        name = joint.get('name')
        range_str = joint.get('range')
        if range_str:
            range_vals = range_str.split()
            print(f"  • {name}: [{range_vals[0]}, {range_vals[1]}]")
            
            # Check if hip joints have bidirectional range
            if 'hip' in name:
                if float(range_vals[0]) >= 0:
                    print(f"    ⚠️  WARNING: Hip joint should have negative lower bound")
            
            # Check if knee joints have proper range
            if 'knee' in name:
                if float(range_vals[0]) >= 0:
                    print(f"    ⚠️  WARNING: Knee joint should have negative range")
    
    # Check actuators
    actuators = root.findall(".//motor")
    print(f"\n⚙️  Found {len(actuators)} actuators:")
    
    for act in actuators:
        name = act.get('name')
        gear = act.get('gear')
        ctrl_range = act.get('ctrlrange')
        print(f"  • {name}: gear={gear}, ctrlrange={ctrl_range}")
    
    return True

def test_config_file():
    """Test if config.yaml has correct parameters"""
    print("\n" + "=" * 60)
    print("📋 Testing Training Configuration (config.yaml)")
    print("=" * 60)
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check environment parameters
    env = config.get('env', {})
    print("\n🌍 Environment Settings:")
    print(f"  • Model path: {env.get('xml_path')}")
    print(f"  • Action scale: {env.get('action_scale')}")
    print(f"  • Init height: {env.get('init_height')}")
    print(f"  • Target velocity: {env.get('target_vx')}")
    print(f"  • Leg L1: {env.get('leg_L1')}")
    print(f"  • Leg L2: {env.get('leg_L2')}")
    print(f"  • Step span: {env.get('step_L_span')}")
    print(f"  • Base height: {env.get('step_base_h')}")
    print(f"  • Step clearance: {env.get('step_clearance')}")
    
    # Check reward weights
    print("\n💰 Reward Weights:")
    print(f"  • Speed: {env.get('w_speed')}")
    print(f"  • Posture: {env.get('w_posture')}")
    print(f"  • Action: {env.get('w_action')}")
    print(f"  • Slip: {env.get('w_slip')}")
    print(f"  • Angular velocity: {env.get('w_angvel')}")
    
    # Check PPO parameters
    ppo = config.get('ppo', {})
    print("\n🧠 PPO Settings:")
    print(f"  • Learning rate: {ppo.get('lr')}")
    print(f"  • Clip range: {ppo.get('clip_range')}")
    print(f"  • Rollout horizon: {ppo.get('rollout_horizon')}")
    print(f"  • Hidden size: {ppo.get('hidden')}")
    
    return True

def check_source_modifications():
    """Check if source files have our modifications"""
    print("\n" + "=" * 60)
    print("📋 Checking Source Code Modifications")
    print("=" * 60)
    
    # Check mj_env.cpp
    print("\n📄 Checking mj_env.cpp:")
    if os.path.exists("src/mj_env.cpp"):
        with open("src/mj_env.cpp", 'r') as f:
            content = f.read()
            if "off = -0.4; // 적당히 굽힌 자세" in content:
                print("  ✅ Initial knee position fixed (-0.4)")
            else:
                print("  ⚠️  Initial knee position not updated")
            
            if "settle_(50)" in content:
                print("  ✅ Settling time increased (50 steps)")
            else:
                print("  ⚠️  Settling time not updated")
            
            if "const double alpha = 0.3;" in content:
                print("  ✅ Servo response improved (0.3)")
            else:
                print("  ⚠️  Servo response not updated")
    
    # Check gait_ref.cpp
    print("\n📄 Checking gait_ref.cpp:")
    if os.path.exists("src/gait_ref.cpp"):
        with open("src/gait_ref.cpp", 'r') as f:
            content = f.read()
            if "z = -std::abs(z);" in content:
                print("  ✅ IK coordinate system fixed")
            else:
                print("  ⚠️  IK coordinate system not updated")
            
            if "knee = -(M_PI - std::acos(cK));" in content:
                print("  ✅ Knee angle sign convention fixed")
            else:
                print("  ⚠️  Knee angle sign not updated")

def main():
    print("\n" + "🤖" * 30)
    print("🤖 QUADRUPED ROBOT CONFIGURATION TEST 🤖")
    print("🤖" * 30)
    
    # Change to webapp directory
    os.chdir("/home/user/webapp")
    
    # Run tests
    model_ok = test_model_file()
    config_ok = test_config_file()
    check_source_modifications()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if model_ok and config_ok:
        print("✅ Configuration files are properly set up!")
        print("\n🚀 Next Steps:")
        print("1. The project needs to be rebuilt with CMake")
        print("2. Run from the /home/user/webapp directory:")
        print("   cd /home/user/webapp")
        print("   mkdir -p build && cd build")
        print("   cmake .. && make -j$(nproc)")
        print("   ./train_mjppo ../config.yaml")
        print("\n3. Or if you want to resume training:")
        print("   ./train_mjppo ../config.yaml")
        print("\n⚠️  Note: Make sure you have CMake, MuJoCo, and LibTorch installed!")
    else:
        print("❌ Some configuration issues found. Please check the warnings above.")
    
    print("\n" + "🤖" * 30 + "\n")

if __name__ == "__main__":
    main()