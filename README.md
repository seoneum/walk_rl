# 🤖 Quadruped Robot RL Walking Simulation - Fixed Version

## 🔧 Problem Solved
The quadruped robot was jumping instead of walking due to several configuration issues. This has been fixed!

## ✅ Applied Fixes

### 1. **Joint Range Configuration (models/quadruped.xml)**
- ✅ Hip joints: Now have bidirectional movement `[-0.5, 0.5]`
- ✅ Knee joints: Proper bending range `[-2.0, 0]`
- ✅ Actuator gear: Reduced from 15 to 10 for smoother control
- ✅ Control range: Adjusted for bidirectional control

### 2. **IK Coordinate System (src/gait_ref.cpp)**
- ✅ Fixed z-axis direction for MuJoCo coordinate system
- ✅ Corrected knee angle sign convention
- ✅ Removed redundant sign flipping

### 3. **Initial Posture (src/mj_env.cpp)**
- ✅ Knee initial position: -0.4 (less crouched)
- ✅ Hip initial position: 0.0 (neutral)
- ✅ Settling time: 50 steps
- ✅ Servo response: α = 0.3

### 4. **Training Parameters (config.yaml)**
- ✅ Action scale: 0.20
- ✅ Initial height: 0.25m
- ✅ Leg geometry: L1=0.30m, L2=0.30m
- ✅ Enhanced posture reward: 0.5
- ✅ Learning rate: 1e-4
- ✅ Rollout horizon: 1024

## 🚀 How to Build and Run

### Prerequisites
- CMake (3.10+)
- MuJoCo
- LibTorch (PyTorch C++)
- CUDA (optional)

### Build Instructions

```bash
# 1. Navigate to the project directory
cd /home/user/webapp

# 2. Create and enter build directory
mkdir -p build && cd build

# 3. Configure with CMake
cmake ..

# 4. Build the project
make -j$(nproc)
```

### Running Training

#### Start New Training
```bash
cd /home/user/webapp/build
./train_mjppo ../config.yaml
```

#### Resume Training from Checkpoint
```bash
cd /home/user/webapp/build
./train_mjppo ../config.yaml ../runs/[run_folder]/ckpt_[epoch].pt
```

#### View Robot in Simulator
```bash
cd /home/user/webapp/build
./quad_viewer ../config.yaml [optional_checkpoint.pt]
```

## 📁 Project Structure

```
/home/user/webapp/
├── config.yaml          # Training configuration
├── models/
│   └── quadruped.xml   # Robot model (FIXED)
├── src/
│   ├── mj_env.cpp      # Environment (FIXED)
│   ├── gait_ref.cpp    # Gait reference (FIXED)
│   ├── train.cpp       # Training script
│   └── viewer.cpp      # Visualization
├── include/            # Header files
├── runs/              # Training outputs
└── build/            # Build directory (create this)
```

## 🧪 Testing Your Setup

Run the test script to verify all fixes are applied:
```bash
cd /home/user/webapp
python3 test_robot.py
```

## 📊 Expected Results

After applying these fixes, you should see:
- ✅ Robot walks smoothly without jumping
- ✅ Stable training convergence
- ✅ Gradually increasing rewards
- ✅ Natural gait emergence

## ⚠️ Common Issues

### "Failed to load config YAML"
- Make sure you run from the build directory
- Provide the correct path to config.yaml: `../config.yaml`

### "Model file not found"
- Ensure you're in the correct directory
- The model path in config.yaml should be: `models/quadruped.xml`

### Build Errors
- Install missing dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install cmake build-essential
  
  # Install MuJoCo
  # Follow instructions at: https://mujoco.org/
  
  # Install LibTorch
  # Download from: https://pytorch.org/get-started/locally/
  ```

## 📈 Training Tips

1. **Monitor the rewards**: They should gradually increase
2. **Check robot behavior**: Use the viewer to ensure no jumping
3. **Adjust parameters if needed**: 
   - Reduce `action_scale` if movements are too aggressive
   - Increase `w_posture` if robot is unstable
   - Decrease learning rate if training is unstable

## 🎯 Pull Request

The fixes have been submitted as a PR: https://github.com/seoneum/walk_rl/pull/1

## 📝 Citation

If you use this fixed version, please mention the fixes applied to resolve the jumping issue.

---
**Fixed by**: GenSpark AI Developer
**Date**: 2025-08-18
**Issue**: Quadruped robot jumping instead of walking in RL simulation