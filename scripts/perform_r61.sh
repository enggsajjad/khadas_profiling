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
#docker run --rm --name conv-test -it  --mount type=bind,source=${PDIR}/convertdemo,target=/acuity-toolkit/convertdemo --entrypoint=/acuity-toolkit/convertdemo/network/convert-mnist-onnx-to-khadas.sh  ghcr.io/scholz/aml-container:0.0.1
echo -e "\e[35m>>>>> Executing the Conversion Scripts Done! <<<<<<<\e[37m"


#----------------------------- REMOTE KHADAS -----------------------------

#--------------------------------- TESTS ---------------------------------
# Test, whether the needed environments have a realistic (!) value.

test -d ${CONV_SCRIPTS_PATH} || {
  echo
  echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE CONV_SCRIPTS_PATH \"${CONV_SCRIPTS_PATH}\" DOES NOT EXIST!"
  echo "       MAYBE IN THE export settings SOMETHING IS WRONG."
  echo
  exit 1
}

#test -d ${WORKFLOW_PROJECT} || {
#  echo
#  echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE WORKFLOW_PROJECT \"${WORKFLOW_PROJECT}\" DOES NOT EXIST!"
#  echo "       MAYBE IN THE export settings SOMETHING IS WRONG."
#  echo
#  exit 1
#}

#--------------------------------- REMOTE EXECUTIONS ---------------------------------

echo -e "\e[35m>>>>> Creating Remote NBG Directory... <<<<<<<\e[37m"
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR} && mkdir -p ${NBG_DIR} )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR} NOT FOUND OR ${NBG_DIR} CAN'T BE CREATED !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Creating Remote NBG Directory Done! <<<<<<<\e[37m"

echo -e "\e[35m>>>>> Copying C,H & NB files to Remote NBG Directory... <<<<<<<\e[37m"
scp -r ${CONV_SCRIPTS_PATH}/${CONV_NBG_FOLDER}/*.c  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE C FILES COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
scp -r ${CONV_SCRIPTS_PATH}/${CONV_NBG_FOLDER}/*.h  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE HEADER FILES COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
scp -r ${CONV_SCRIPTS_PATH}/${CONV_NBG_FOLDER}/${NN_NAME}.nb  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE ${NN_NAME}.NB FILE COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Copying C,H & NB files to Remote NBG Directory Done! <<<<<<<\e[37m"

echo -e "\e[35m>>>>> Copying Template Makefile & Build files to Remote NBG Directory... <<<<<<<\e[37m"
scp -r ${WORKFLOW_PROJECT}/${NBG_TEMPLATE_SCRIPTS}/*.*  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE NBG DEFAULT SCRIPTS COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && echo "TARGET_NAME = ${NN_NAME}" > makefile.target_name )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR} NOT FOUND OR makefile.target_name CAN'T BE ALTERED !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Copying Template Makefile & Build files to Remote NBG Directory Done! <<<<<<<\e[37m"

ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && patch -N main.c < main.patch )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE main.patch NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Patching main.c on Remote NBG Directory Done! <<<<<<<\e[37m"

echo -e "\e[35m>>>>> Compiling the model on Remote Khadas... <<<<<<<\e[37m"
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && bash ./build_vx.sh )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Compiling the model on Remote Khadas Done! <<<<<<<\e[37m"

echo -e "\e[35m>>>>> Copying test input image to Remote... <<<<<<<\e[37m"
scp -r ${INPUT_IMAGE}  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/testimg.jpg || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE ${INPUT_IMAGE} FILE COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Copying test input image to Remote! <<<<<<<\e[37m"


echo -e "\e[35m>>>>> Executing the model on Remote Khadas... <<<<<<<\e[37m"
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && ./bin_r_cv4/${NN_NAME} ${NN_NAME}.nb ./testimg.jpg "$1")" > ${LOGFILE}  || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR ./bin_r_cv4/${NN_NAME} CAN'T BE RUN !!"
	echo
	exit 1
  }
echo -e "\e[35m>>>>> Executing the model on Remote Khadas Done! <<<<<<<\e[37m"
echo -e "\e[35m>>>>> Executing network on Khadas is saved in model_execution.log <<<<<<<\e[37m"

