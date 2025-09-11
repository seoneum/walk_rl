#!/bin/bash

echo "======================================"
echo "🔧 QUADRUPED JUMPING FIX SCRIPT"
echo "======================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}1. Checking current branch...${NC}"
git branch --show-current

echo -e "\n${YELLOW}2. Analyzing model file...${NC}"
echo "Joint ranges in quadruped.xml:"
grep -E "joint.*range=" models/quadruped.xml | head -8

echo -e "\n${YELLOW}3. Checking actuator settings...${NC}"
grep -E "motor.*gear=" models/quadruped.xml | head -4

echo -e "\n${YELLOW}4. Checking initial height in configs...${NC}"
echo "config.yaml:"
grep "init_height" config.yaml
echo "stable_config.yaml:"
grep "init_height" stable_config.yaml

echo -e "\n${YELLOW}5. Key modifications needed:${NC}"
echo -e "${GREEN}✓${NC} Reduced initial torso height: 0.7m → 0.3m"
echo -e "${GREEN}✓${NC} Reduced actuator gear: 15 → 5"
echo -e "${GREEN}✓${NC} Increased joint damping: 0.1 → 0.5"
echo -e "${GREEN}✓${NC} Reduced position KP: 120 → 50"
echo -e "${GREEN}✓${NC} Fixed foot collision geometry"
echo -e "${GREEN}✓${NC} Very low initial height: 0.12m"
echo -e "${GREEN}✓${NC} Minimal action scale: 0.05"

echo -e "\n${YELLOW}6. Build commands:${NC}"
echo "mkdir -p build && cd build"
echo "cmake .."
echo "make -j\$(nproc)"

echo -e "\n${YELLOW}7. Test with stable config:${NC}"
echo "cd build"
echo "./train_mjppo ../stable_config.yaml"

echo -e "\n${RED}CRITICAL NOTES:${NC}"
echo "- If robot still jumps, the issue might be in the compiled binary"
echo "- Make sure to rebuild after XML changes"
echo "- Use stable_config.yaml for initial testing"
echo "- Monitor z-position in first few steps"

echo -e "\n======================================"