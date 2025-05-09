#!/bin/bash
# Purpose: One-click run to initialize the Jupyter Lab work environment for Automated Self-checkout with OpenVino Toolkit
# Ref.Imp. Author: https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/automated_self_checkout
# Script: Mario Divan
# ------------------------------------------
source utilities.sh

# Script Data
venvironment="venv" #Virtual Environment name

if [[ -n $1 ]]; then
	venvironment=$1
fi

counter=0
for idx in "${packages[@]}"; do
	if isInstalled "${idx}" -ge 1 &>/dev/null; then
		counter=$((counter + 1))
	else
		mess_err "${idx} could not be installed"
		exit 99
	fi
done

mess_inf "Checking the OpenVino Toolkit Installation..."
if [[ ! -d "openvino_build_deploy" ]]; then
	mess_err "The openvino_build_deploy folder is not available. Please run the installEnv.sh script first"
	exit 99
else
	mess_oki "The openvino_build_deploy folder has been detected"
fi

mess_inf "Entering into the openvino_build_deploy/ai_ref_kits/automated_self_checkout folder..."
cd openvino_build_deploy/ai_ref_kits/automated_self_checkout || exit 99
if [[ $? -eq 0 ]]; then
	mess_inf "The Self-Checkout AI reference kit is available"
else
	mess_err "The Self-Checkout AI reference kit is unavailable"
fi

mess_inf "Activating the Virtual Environment (${venvironment})"
source "${venvironment}/bin/activate"

if [[ $? -eq 0 ]]; then
	mess_oki "Virtual environment (${venvironment}) activated successfully!"
else
	mess_err "Error activating the virtual environment (${venvironment})."
	exit 99
fi

jupyter lab self-checkout-recipe.ipynb

deactivate

if [[ $? -eq 0 ]]; then
	mess_oki "The Virtual environment (${venvironment})  has been deactivated!"
else
	mess_err "The virtual environment (${venvironment}) coud not be deactivated."
	exit 99
fi
