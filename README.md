# walk_rl

Experimental quadruped locomotion project in C++17 using MuJoCo, LibTorch, PPO/ARS, and YAML configuration.

## Status

This repository is a research prototype. The source tree includes a MuJoCo environment, PPO and ARS implementations, gait references, configuration loading, training, and visualization tools.

An existing pull request contains behavioral changes intended to address a jumping issue. This cleanup is limited to repository hygiene and does not choose or merge those algorithmic changes.

## Source layout

- `include/`: public C++ interfaces
- `src/`: environment, algorithms, training, and viewer code
- `models/`: MuJoCo model assets
- `tools/`: project utilities
- `config.yaml`: experiment configuration

## Build

Install MuJoCo, LibTorch, yaml-cpp, CMake, and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build --config Release
```

Dependency locations should be supplied through CMake or environment variables.

## Repository policy

Build trees, ROS install/log directories, training runs, checkpoints, and generated screenshots are excluded from Git. Store large experiment outputs outside the source repository.

> Removing generated files from the current tree does not erase them from earlier Git history.
