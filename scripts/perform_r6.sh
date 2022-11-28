#!/bin/bash
loops="$1"
INPUT_IMAGE="$2"
LOGFILE="$3"
######################################################################
#
# SETTINGS ON HOST UBUNTU.
#
######################################################################
source ../scripts/env_settings_r2.sh
echo -e "\e[35m>>>>> Setting Environment Variables Done! <<<<<<<\e[37m"
#----------------------------- UBUNTU HOST -----------------------------

# Test, whether the needed environments have a realistic (!) value.
test -f ${CONV_SCRIPTS_PATH}/network/${NN_MODEL} || {
  echo
  echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE  \"${NN_MODEL}\" DOES NOT EXIST!"
  echo "       MAYBE IN THE export settings SOMETHING IS WRONG."
  echo
  exit 1
}

echo -e "\e[35m>>>>> Copying Quantization Images and Model... <<<<<<<\e[37m"
#cp ${NOTEBOOK_PATH}/${NN_MODEL} ${CONV_SCRIPTS_PATH}/network/.

## cp quantization data
cp -r ../quantization_images/* ${CONV_SCRIPTS_PATH}/dataset/.
#cp -r ./quantization_images/* ${CONV_SCRIPTS_PATH}/data/bmp/.
#cp -r ./example_images_rgb ${CONV_SCRIPTS_PATH}/dataset/.
_curdir=$PWD
cd ${CONV_SCRIPTS_PATH}/dataset
ls *.bmp -1 > dataset1.txt
cd $_curdir
echo -e "\e[35m>>>>> Copying Quantization Images and Model Done! <<<<<<<\e[37m"


echo -e "\e[35m>>>>> Executing the Conversion Scripts... <<<<<<<\e[37m"
#cd ${CONV_SCRIPTS_PATH}
#bash 0_import_model_mnist.sh && bash 1_quantize_model_mnist.sh && bash 2_export_case_code_mnist.sh
#cd -
PDIR="$(dirname "${PWD}")"

