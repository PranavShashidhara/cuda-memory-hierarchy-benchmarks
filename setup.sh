#!/bin/bash
 
# setup.sh - Initialize folder structure for cuda_bench repository
# This script creates all necessary directories (preserves existing files)
 
set -e  # Exit on error
 
# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
 
echo -e "${GREEN}Setting up cuda_bench repository structure...${NC}"
 
# Create main directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p baseline/results
mkdir -p Optimized/results
mkdir -p build
mkdir -p results
 
echo ""
echo -e "${GREEN}✓ Directory structure created successfully!${NC}"
echo ""
echo "Directory tree:"
tree -L 2 2>/dev/null || find . -type d | sort | sed 's|[^/]*/| |g'
echo ""
 