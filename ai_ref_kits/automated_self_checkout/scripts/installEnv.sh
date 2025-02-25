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

total=${#packages[*]} #array length
counter=0
mess_inf "Total Required Packages to Verify: ${total}"

for idx in "${packages[@]}"; do
	sudo apt install "$idx" -y
	
	if isInstalled "$idx"&>/dev/null -ge 1; then
		counter=$((counter+1))
	else
		mess_err "$idx could not be installed"
		exit 99
	fi
done

for idx in "${packages[@]}"; do
	sudo dpkg-query -W -f='${Package} ${Version}. Status: ${Status}\n' "${idx}"
done

if [[ ! -d "openvino_build_deploy" ]]; then
	git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
fi

if [[ ! -d "openvino_build_deploy" ]]; then
	mess_err "The openvino_build_deploy folder could not be created"
	exit 99
else
	mess_oki "The openvino_build_deploy folder has been created"
fi

# Move the remaining scripts to the current installEnv.sh location
if [[ -d "openvino_build_deploy/ai_ref_kits/automated_self_checkout/scripts" ]]; then
	for ovscript in "${ovscripts[@]}"; do
		
		cp openvino_build_deploy/ai_ref_kits/automated_self_checkout/scripts/$ovscript .
		if [[ $? -eq 0 ]]; then
			mess_oki "The $ovscript script has been copied"
		else
			mess_war "The $ovscript script could not be copied"
		fi
	done
else
	mess_war "The openvino_build_deploy/ai_ref_kits/automated_self_checkout/scripts folder does not exist"
fi

mess_inf "Entering into the openvino_build_deploy/ai_ref_kits/automated_self_checkout folder..."
cd openvino_build_deploy/ai_ref_kits/automated_self_checkout || exit

mess_inf "Discarding any local change..."
git checkout .
mess_inf "Pulling the openvino repository..."
git pull
mess_inf "Fetching the openvino repository..."
git fetch

git lfs -X= -I=data/ pull

if [[ $? -eq 0 ]]; then
	mess_oki "Video sample pulled."
else
	mess_err "The Video sample has not been pulled."
	exit 99
fi

mess_inf "Creating a virtual environment $venvironment (If it exists, it will delete the pre-existent ) ..."
python3 -m venv "$venvironment" #--clear

if [[ $? -eq 0 ]]; then
	mess_oki "Virtual environment ($venvironment) created successfully!"
else
	mess_err "Error creating the virtual environment ($venvironment)."
	exit 99
fi

mess_inf "Activating the Virtual Environment ($venvironment)"
source "$venvironment/bin/activate"

if [[ $? -eq 0 ]]; then
	mess_oki "Virtual environment ($venvironment) activated successfully!"
else
	mess_err "Error activating the virtual environment ($venvironment)."
	exit 99
fi

python -m pip install --upgrade pip

echo "lapx>=0.5.2" >>requirements.txt #Adding lapx library to the requirements.txt file (It was missing but required)
echo "spaces>=0.3.2" >>requirements.txt #Adding spaces library to the requirements.txt file (It was missing but required Gradio)
echo "gradio>=5.16.0" >>requirements.txt #Adding spaces library to the requirements.txt file (It was missing but required Gradio)
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
	mess_oki "Packages have been installed in the Virtual environment ($venvironment)  successfully!"
else
	mess_err "Error installing required packages in the the virtual environment ($venvironment)."
	exit 99
fi

mess_oki "Your virtual environment ($venvironment) is ready to run the Self-checkout application."

mess_inf "Starting Jupyter Lab..."

jupyter lab self-checkout-recipe.ipynb

deactivate

if [[ $? -eq 0 ]]; then
	mess_oki "The Virtual environment ($venvironment)  has been deactivated!"
else
	mess_err "The virtual environment ($venvironment) coud not be deactivated."
	exit 99
fi
