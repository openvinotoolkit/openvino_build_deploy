#!/bin/bash
# Purpose: Utility functions for scripting and common variables (e.g., packages to manage)
# Script: Mario Divan
# ------------------------------------------

RED='\033[0;31m'    # Red
BLUE='\033[0;34m'   # Blue
CYAN='\033[0;36m'   # Cyan
GREEN='\033[0;32m'  # Green
YELLOW='\033[0;33m' # Yellow
NOCOLOR='\033[0m'
BWHITE='\033[1;37m' # White

mess_err() {
	printf "${RED}\u274c ${BWHITE} $1\n"
}

mess_oki() {
	printf "${GREEN}\u2705 ${NOCOLOR} $1\n"
}

mess_war() {
	printf "${YELLOW}\u26A0  ${BWHITE} $1\n"
}

mess_inf() {
	printf "${CYAN}\u24d8  ${NOCOLOR} $1\n"
}

isInstalled() {
	mess_inf "Verifying $1 package"
	return dpkg-query -Wf'${Status}' $1 2>/dev/null | grep 'ok installed' | wc -l
}

declare -a packages=("git" "git-lfs" "gcc" "python3-venv" "python3-dev" "ffmpeg")
declare -a ovscripts=("utilities.sh" "installEnv.sh" "runEnv.sh" "runDemo.sh" "cleanEnv.sh")
