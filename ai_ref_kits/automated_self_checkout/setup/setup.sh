#!/bin/bash
# Purpose: One-click installator for Automated Self-checkout with OpenVino Toolkit
# Ref.Imp. Author: https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/automated_self_checkout
# Script: Mario Divan
# ------------------------------------------

cd ~

if [[ ! $? -eq 0 ]]; then
	echo "Failed to change to home directory"
	exit 1
fi

if [[ ! -d "oneclickai" ]]; then
	mkdir oneclickai
fi

cd oneclickai

if [[ ! $? -eq 0 ]]; then
	echo "Failed to change directory to ~/oneclickai"
	exit 1
fi

sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install wget
sudo apt autoremove -y

wget https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/automated_self_checkout/setup/utilities.sh -O utilities.sh

wget https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/ai_ref_kits/automated_self_checkout/setup/installEnv.sh -O installEnv.sh

sudo chmod +x utilities.sh
sudo chmod +x installEnv.sh

if [[ -f "utilities.sh" && -f "installEnv.sh" ]]; then
	./installEnv.sh
else
	echo "utilities.sh or installEnv.sh not found!"
	exit 1
fi
# ------------------------------------------
