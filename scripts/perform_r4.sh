#!/bin/bash

######################################################################
#
# SETTINGS ON HOST UBUNTU.
#
######################################################################
source ./env_settings.sh

#----------------------------- UBUNTU HOST -----------------------------

# Test, whether the needed environments have a realistic (!) value.
test -f ${NOTEBOOK_PATH}/${NN_MODEL} || {
  echo
  echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE  \"${NN_MODEL}.onnx\" DOES NOT EXIST!"
  echo "       MAYBE IN THE export settings SOMETHING IS WRONG."
  echo
  exit 1
}

cp ${NOTEBOOK_PATH}/${NN_MODEL} ${CONV_SCRIPTS_PATH}/network/.
echo "Executing the Conversion Scripts..."
cd ${CONV_SCRIPTS_PATH}
bash 0_import_model_mnist.sh && bash 1_quantize_model_mnist.sh && bash 2_export_case_code_mnist.sh
cd -
echo "Executing the Conversion Scripts Done!"


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

test -d ${WORKFLOW_PROJECT} || {
  echo
  echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE WORKFLOW_PROJECT \"${WORKFLOW_PROJECT}\" DOES NOT EXIST!"
  echo "       MAYBE IN THE export settings SOMETHING IS WRONG."
  echo
  exit 1
}

#--------------------------------- REMOTE EXECUTIONS ---------------------------------

ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR} && mkdir -p ${NBG_DIR} )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR} NOT FOUND OR ${NBG_DIR} CAN'T BE CREATED !!"
	echo
	exit 1
  }
  
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


scp -r ${WORKFLOW_PROJECT}/${NBG_SCRIPTS}/*.*  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${NBG_DIR}/. || {
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

ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && patch -N main.c < main.patch )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE main.patch NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }

ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && bash ./build_vx.sh )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }

echo "Executing network on Khadas."
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && ./bin_r_cv4/${NN_NAME} ${NN_NAME}.nb ./0.bmp.jpg "$1")" > model_execution.log  || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR ./bin_r_cv4/${NN_NAME} CAN'T BE RUN !!"
	echo
	exit 1
  }
echo "Executing network on Khadas is saved in model_execution.log"



