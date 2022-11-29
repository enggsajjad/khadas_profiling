# README #

AUTOMATION FLOW

### What is this repository for?
* implements the complete flow on ubuntu host for executing and profiling the converted model on khadas npu
* parses the results fetched from the khadas for timing informations
* utilizes the docker container to run conversion script with sdk 4.6.6.2 version

### What should be installed before executing notebook?
Make sure that:

1. Docker is installed
2. Visual Studio Code and Python, Jupyter Extensions are installed.
3. IMPORTANT!!! SSH files/passwords are copied to Khadas for a user.

### Requirements
Pleaase install the following:

1. python --version Python 3.10.6
2. python -m pip install onnx matplotlib numpy onnx onnxruntime opencv-python Pillow ipykernel torch torchaudio torchvision nni pandas tensorboard docker pgi
3. sudo apt-get install python-setuptools python-dev build-essential
4. sudo apt install docker-ce
https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket
https://cloudcone.com/docs/article/how-to-install-docker-on-ubuntu-22-04-20-04/
5. sudo groupadd docker
6. sudo usermod -aG docker $USER
7. sudo chmod 666 /var/run/docker.sock
8. Software Center Installations: Visual Studio Code and Notepad++
9. VSCode Extensions: Python, notebook for python kernel
10. add ssh key to remote khadas system, using:
ssh-keygen
ssh-copy-id khadas@10.10.254.119
11. Check that all the .sh script are set with executable attributes (chmod +x)


### How to run?:
* change directory (CD) to the notebook folder
* open the jupyter notebook from the notebook folder and run it. this will implement all the conversion, export, copy to khadas, compiling, and executing there and producing the profiling times in the array.
Things to set in the Notebook:
1. Test Input image
2. Image sizes (channel, height & width)
3. model to be profiled

##   Main Contents:
1. "notebook" folder that contains the jupyter notebooks
2. "convertdemo" folder contains the docker container, dataset folder, network folder, and conversion scripts
3. "template_nbg_unify" folder that contains the default makefiles abd build script for compiling a network
4. "template_normal_case" folder that contains the default makefiles abd build script for compiling a network for profiling (not available now)
5. "scripts" folder contains different shell scripts
6. "logs" contains the logs generated from the profiling
7. "quantization_images" contains to store and then use the images while quantization step of conversion scripts. these images can also be used for testing
8. "example_images_rgb" some example images might be used for the tests  (not available now)

###   Notebook Contents:
1. SimpleProfile.ipynb for random image size
2. MnistProfile.ipynb for mnist dataset   (not available now)

###   Convertdemo Contents:
1. dataset folder: contains the quantization images to be used for conversion script
2. network folder: contains the shell script to execute conversion script and model onnx file to input to conversion script
3. nbg_unify_mnist: the folder for generated/converted code
4. 0_import_model_mnist.sh: Conversion Scripts for importing the model
5. 1_quantize_model_mnist.sh:  Conversion Scripts for quantizing the model
6. 2_export_case_code_mnist.sh Conversion Scripts for exporting the converted model

###   Scripts Contents:
1. "perform_test.sh" performs the complete flow from host to khadas and get back the results
2. "parse_r1.sh" parses the results fetched from the khadas for timing informations
3. "env_settings_r2.sh" contains different environment variables to be used for other scripts

##   Perform.sh:

Parameters: number of loop, test input image, lgo file

sajjad@teco:~/sajjad/scripts/notebook$ ../scripts/perform_r6.sh 10 ../convertdemo/dataset/mnist2.jpg ../logs/model_execution.log
#### Steps:

1. clone the repository on the host
2. you should have already added host ssh-key into remote khadas and vice-versa.
3. set the required variables in the script "perform.sh"; go through all the variables, the comments explain each of them.
4. check the path of the source (host) and destination(remote) in the script "perform.sh"
5. run the "perform.sh" from host. it copy required .h, .c, .nb, .export.data, to the khadas, compile the network on the khadas, execute there and profile there also.
6. the log for the executing network on the khadas and corresponding profiling will be logged in the "model_execution.log" and "model_profiling.log" on the host.

##   Parse.sh:

Parameters: input file, output file, and number of loops for executing

Parse the results fetched from the Khadas.

1. run the "parse.sh" on the results/logs taken from the execution/profiling on the khadas.
2. takes one input parameter: the file to be parsed; one output parameter: the parsed output file
