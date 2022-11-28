#!/bin/bash

######################################################################
#
# SETTINGS ON HOST UBUNTU.
#
######################################################################


#----------------------------- UBUNTU HOST -----------------------------
# Ubuntu Host Adreess
export HOST_ADDRESS=10.10.254.178
# Ubuntu Host User
export HOST_USER=sajjad
# the path for the conversion scripts folder
export CONV_SCRIPTS_PATH=/home/sajjad/sajjad/npusdk2_4662/aml_npu_sdk/acuity-toolkit/demo
# the folder where NBG conversion files are created
export CONV_NBG_FOLDER=nbg_unify_mnist
# the NN model appliction name
export NN_NAME=mnist
# the NN model appliction onnx files
export NN_MODEL=${NN_NAME}.onnx
# the path of the folders where default workflow scripts nbg_unify and normal_case makefile/build scripts are located
export WORKFLOW_PROJECT=/home/sajjad/sajjad/scripts
# the folders where default nbg_unify makefile/build scripts are located
export NBG_SCRIPTS=nbg_unify


#----------------------------- REMOTE KHADAS -----------------------------
# Remote Khadas Adreess
export KHADAS_ADDRESS=10.10.254.119
# Remote Khadas User
export KHADAS_USER=khadas
# the path on remote khadas where NBG conversion folder is to be created and executed
export REMOTE_PROJECT_DIR=/home/khadas/hussain
# the path on remote khadas where NBG conversion folder is to be created and executed
export NBG_DIR=nbg_unify_mnist

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


ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && bash ./build_vx.sh )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }

echo "Executing network on Khadas."
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${NBG_DIR} && ./bin_r_cv4/${NN_NAME} ${NN_NAME}.nb ./0.bmp.jpg )" > model_execution.log  || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${NBG_DIR} NOT FOUND OR ./bin_r_cv4/${NN_NAME} CAN'T BE RUN !!"
	echo
	exit 1
  }
echo "Executing network on Khadas is saved in model_execution.log"

######################################################################
#
# PROFILING SETTINGS
#
######################################################################
# the path on remote khadas where profile conversion folder is to be created and executed
export PROFIL_DIR=op_test_mnist_normal_case_demo
# the folder where Normal Case conversion files are created
export CONV_CASE_FOLDER=mnist_normal_case_demo
# the folders where default normal_case makefile/build scripts are located
export PROFILE_SCRIPTS=normal_case

#--------------------------------- REMOTE EXECUTIONS ---------------------------------

ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR} && mkdir -p ${PROFIL_DIR} )" || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR} NOT FOUND OR ${PROFIL_DIR} CAN'T BE CREATED !!"
	echo
	exit 1
  }
  
scp -r ${CONV_SCRIPTS_PATH}/${CONV_CASE_FOLDER}/*.c  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${PROFIL_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE C FILES COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
scp -r ${CONV_SCRIPTS_PATH}/${CONV_CASE_FOLDER}/*.h  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${PROFIL_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE HEADER FILES COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
scp -r ${CONV_SCRIPTS_PATH}/${CONV_CASE_FOLDER}/${NN_NAME}.export.data  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${PROFIL_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE ${NN_NAME}.EXPORT.data FILES COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }



scp -r ${WORKFLOW_PROJECT}/${PROFILE_SCRIPTS}/*.*  ${KHADAS_USER}@${KHADAS_ADDRESS}:${REMOTE_PROJECT_DIR}/${PROFIL_DIR}/. || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE DEFAULT PROFILING SCRIPTS COULDN'T BE COPIED TO ${KHADAS_ADDRESS} !!"
	echo
	exit 1
  }
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${PROFIL_DIR} && echo "TARGET_NAME = ${NN_NAME}" > makefile.target_name )"  || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR} NOT FOUND OR makefile.target_name CAN'T BE ALTERED !!"
	echo
	exit 1
  }


ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "( cd ${REMOTE_PROJECT_DIR}/${PROFIL_DIR} && bash ./build_vx.sh )"  || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${PROFIL_DIR} NOT FOUND OR build_vx.sh CAN'T BE EXECUTED !!"
	echo
	exit 1
  }

echo "Profiling network on Khadas."
ssh ${KHADAS_USER}@${KHADAS_ADDRESS} "(source ${REMOTE_PROJECT_DIR}/${PROFIL_DIR}/profile_exports.sh && cd ${REMOTE_PROJECT_DIR}/${PROFIL_DIR} && ./bin_r_cv4/${NN_NAME} ${NN_NAME}.export.data ./0.bmp.jpg )" > model_profiling.log   || {
	echo
	echo "ERROR: ABORTING \"perform.sh\", BECAUSE THE REMOTE ${REMOTE_PROJECT_DIR}/${PROFIL_DIR} NOT FOUND OR ./bin_r_cv4/${NN_NAME} CAN'T BE RUN !!"
	echo
	exit 1
  }
echo "Profiling network on Khadas is saved in model_profiling.log"


