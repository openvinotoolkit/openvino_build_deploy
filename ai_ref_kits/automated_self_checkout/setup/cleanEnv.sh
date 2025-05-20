#!/bin/bash
# Purpose: One-click installator for Automated Self-checkout with OpenVino Toolkit
# Ref.Imp. Author: https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/automated_self_checkout
# Script: Mario Divan
# ------------------------------------------
# Script Data
venvironment="venv" #Virtual Environment name

if [[ -n $1 ]]; then
	venvironment=$1
fi
# ------------------------------------------

source utilities.sh

mess_war "This script will remove all the packages required for the Self-checkout AI reference kit."
read -p "Do you want to continue? (y/n): " -n 1 -r
echo "\n"

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    mess_inf "The script has been cancelled."
    exit 99
fi

if [[ -d "openvino_build_deploy" ]]; then
    mess_inf "The openvino_build_deploy folder has been detected."
    sudo rm -rf openvino_build_deploy
else
    mess_war "The openvino_build_deploy folder does not exist."
fi

if [[ -d "model" ]]; then
    mess_inf "The model folder has been detected."
    sudo rm -rf model
fi

mess_oki "Folder, packages, and virtual environment have been successfully removed."