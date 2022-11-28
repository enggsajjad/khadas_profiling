from pickletools import float8
from unicodedata import decimal
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, QAT_Quantizer

import torch.nn as nn
import onnx
import onnx.numpy_helper
from math import ceil
### markus
import numpy as np
import PIL
import os
from re import L
import subprocess
from subprocess import DEVNULL, STDOUT
from xmlrpc.client import boolean
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
#%matplotlib notebook

#imgSizeW= 82#ceil(imgSizeW/4)*4
#imgSizeH= 1#ceil(imgSizeH/4)*4

#gen_img_input_dim_w = imgSizeW
#gen_img_input_dim_h = imgSizeH
#gen_img_input_channels = 3
#test input image
test_input_data="../convertdemo/dataset/rand_3.jpg"
#Paths
quant_image_path = "../quantization_images"
script_path = "../scripts"
log_path = "../logs"
network_path = "../convertdemo/network"
#Files
perform_script = "perform_test.sh"
parse_script = "parse_r1.sh"
perform_log_file = "model_execution.log"
parsed_log_file = "model_execution_parsed.log"
model_name="mnist"


# Generate images based on some arbitrary input dimension
path = quant_image_path

def generate_random_images(xdim, ydim, channels=3, count=1, path="."):
    """
    This functions generates random bmp images to use for quantization given
    a defined dimension
        @xdim   .. width of images
        @ydim   .. height of images
        @count  .. number of images
        @path   .. path of images
    """

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("The new directory quantization_images is created!")

    #delete the pre generated bmp/jpg files
    filelist = [ f for f in os.listdir(quant_image_path) if (f.endswith(".jpg") or f.endswith(".bmp") ) ]
    for f in filelist:
        os.remove(os.path.join(quant_image_path, f))
        
    for c in range(count):
        rnd_img = np.random.randint(low=0,high=255, size=(ydim, xdim, channels),dtype=np.uint8) #imag.transpose((1,2,0)
        imag_tp = np.ascontiguousarray(rnd_img, dtype=np.uint8)

        pil_image = PIL.Image.frombytes('RGB',(xdim, ydim), imag_tp)
        pil_image.save(path + "/rand_"+str(c)+".bmp")
        pil_image.save(path + "/rand_"+str(c)+".jpg")

#export the model
def export_model_to_onnx(model, input_shape=(1,3,28,28), path=model_name+".onnx"):


    dummy_input = torch.randn(input_shape)
    model.to('cpu')
        
    # very important or must leave out - not sure need to test again...
    #traced = torch.jit.trace(model, input_dimension)
    print("------------- Exporting to onnx")
    torch.onnx.export(
                      model, 
                      dummy_input, 
                      path,
                      opset_version=7,
                      verbose=True,
                      export_params=True, 
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None
    )
    
    print("------------- Checking exported model")
    
    # Load the ONNX model
    onnx_model = onnx.load(path)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a Human readable representation of the graph
    #print( onnx.helper.printable_graph(onnx_model.graph) )

##Create a image dimensions configuration file
#with open(network_path+"/"+"imgSize.config", 'w') as f:
#    f.write('imgSize=%d,%d,%d'%(gen_img_input_channels,gen_img_input_dim_h,gen_img_input_dim_w))

#Profiling related functions


def parse_the_results(inp="model_execution.log",
                      out="model_execution_parsed.log",
                      script="parse.sh",
                      loop=1,
                      show=False):
    """
    parse the output of the profiled log
    Parameters
    ----------
    inp : input file
        file to be parsed
    out : output file
        the output parsed file
    scripts : parsing scripts
        shell scripts to be used for parsing
    loop: number of runs
        the loops to run the model in the main.c
    show: show the scripts output
        to sohw or hide the shell script output
    Returns
    -------
    None
    """
    print("------------- Parsing the profiling results...")
    if show == False:
        r = subprocess.run([script, inp, out,loop], stdout=DEVNULL, stderr=STDOUT)
        #!bash {script} {inp}  {out} {loop}
        
    else:
        r = subprocess.run([script, inp, out,loop])
        #!bash {script} {inp}  {out} {loop} > ../logs/jupyter_parse.log
    
    print("------------- Parsing the profiling results done!")

    return r.returncode

def run_profiler(script=script_path+"/"+"perform.sh",loop=1,test_input="test.jpg",lgofile="model_execution_parsed.log",show=False):
    """
    profile the model
    Parameters
    ----------
    scripts : parsing scripts
        shell scripts to be used for parsing
    loop: number of runs
        the loops to run the model in the main.c
    test_input: input image with path
        input image of said dimensions
    lgofile: log file for profiling
        output log file for profiling
    show: show the scripts output
        to sohw or hide the shell script output
    Returns
    -------
    None
    """
    print("------------- Performing the profiling...")
    #subprocess.check_call(["docker","run"])

    if show == False:
        r = subprocess.run([script, loop, test_input,lgofile], stdout=DEVNULL, stderr=STDOUT)
        #!bash {script} {loop}  {test_input} {lgofile}
    else:
        r = subprocess.run([script, loop, test_input,lgofile])
        #!bash {script} {loop}  {test_input} {lgofile} > ../logs/jupyter.log
        #docker run --rm --name conv-test -it  --mount type=bind,source=${PDIR}/convertdemo,target=/acuity-toolkit/convertdemo --entrypoint=/acuity-toolkit/convertdemo/network/convert-mnist-onnx-to-khadas.sh  ghcr.io/scholz/aml-container:0.0.1
        #subprocess.call(["docker run --rm --name conv-test -it  --mount type=bind,source=/home/sajjad/sajjad/scripts/convertdemo,target=/acuity-toolkit/convertdemo --entrypoint=/acuity-toolkit/convertdemo/network/convert-mnist-onnx-to-khadas.sh  ghcr.io/scholz/aml-container:0.0.1"],shell=True)
        #import subprocess
        #cmd = "docker run --rm --name conv-test -it  --mount type=bind,source=/home/sajjad/sajjad/scripts/convertdemo,target=/acuity-toolkit/convertdemo --entrypoint=/acuity-toolkit/convertdemo/network/convert-mnist-onnx-to-khadas.sh  ghcr.io/scholz/aml-container:0.0.1"
        #p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE)
        #out, err = p.communicate()
        #print(out)
        #subprocess.check_call(["../scripts/perform_r61.sh", loop, test_input,lgofile])
    
    print("------------- Performing the profiling done!")

    return r.returncode



def auto_profile(model,
                 loop=1,
                 imgChannel=3,
                 imgDimH=28,
                 imgDimW=28,
                 modelwithPath=model_name+".onnx",
                 testingInput="../convertdemo/dataset/mnist2.jpg",
                 performScript = script_path+"/"+"perform.sh",
                 parseScript = script_path+"/"+"parse.sh",
                 performLogFile = log_path+"/"+"model_execution.log",
                 parsedLogFile = log_path+"/"+"model_execution_parsed.log",
                 debug=False):
    """
    Convert torch model to onnx model and get layer bits config of onnx model.
    Parameters
    ----------
    model : pytorch model
        The model to speedup by quantization
    loop : loop
        the number of loops to run the model on khadas in main.c
    imgChannel: input channels
        input image channel
    imgDimH: input width
        image width
    imgDimW: input height
        image height
    modelwithPath: absolution model path
        the onnx model with path
    testingInput: input image with path
        input image of said dimensions
    performScript: profiling script
        profiling script that implements whole flow
    parseScript: parse script
        parse script which parses the profiling log
    performLogFile: log file for profiling
        output log file for profiling
    parsedLogFile: log file for parsed profiling
        output log file for parsed profiling
    debug : show debugging
        show the debugging output of the scripts
    Returns
    -------
    pandas frame
        contains the execution times (profiled time)
    status
        the error flag indicating the status
    """
    profilingDone = False;
    #export_model_to_onnx(model,input_shape=(1,imgChannel,imgDimH,imgDimW), path=modelwithPath)
    #export_model_to_onnx(model,input_shape=(1,imgChannel,gen_img_input_dim_h,gen_img_input_dim_w), path=modelwithPath)
    export_model_to_onnx(model,input_shape=(1,imgChannel,imgDimH,imgDimW), path=modelwithPath)

    #Create a image dimensions configuration file
    with open(network_path+"/"+"imgSize.config", 'w') as f:
        f.write('imgSize=%d,%d,%d'%(imgChannel,imgDimH,imgDimW))

    #sajjad@teco:~/sajjad/scripts/notebook$ ../scripts/perform_r6.sh 10 ../convertdemo/dataset/mnist2.jpg ../logs/model_execution.log
    r = run_profiler(performScript,loop,testingInput,performLogFile,debug)
    print("Return state run_profiler: ")
    print(r)

    r = parse_the_results(performLogFile,parsedLogFile,parseScript,loop,debug)
    print("Return state parse_the_results: ")
    print(r)

    #read the results into the pandas
    profiledFrames=pd.read_csv(parsedLogFile, sep=':',header = None)
    profiledFrames[2] = profiledFrames[1].str.extract('(\d+)') 
    profiledFrames[2] = profiledFrames[2].astype('float')

    profilingDone = True;
    return profiledFrames,profilingDone


def doProfiling(model,
                 loop=1,
                 imgChannel=3,
                 imgDimH=28,
                 imgDimW=28,
                 debug=False):
    """
    Convert torch model to onnx model and get layer bits config of onnx model.
    Parameters
    ----------
    model : pytorch model
        The model to speedup by quantization
    loop : loop
        the number of loops to run the model on khadas in main.c
    imgChannel: input channels
        input image channel
    imgDimH: input width
        image width
    imgDimW: input height
        image height
    debug : show debugging
        show the debugging output of the scripts
    Returns
    -------
    pandas frame
        contains the execution times (profiled time)
    status
        the error flag indicating the status
    """

    perform_script_abs = script_path+"/"+perform_script
    parse_script_abs = script_path+"/"+parse_script
    perform_log_file_abs = log_path+"/"+perform_log_file
    parsed_log_file_abs = log_path+"/"+parsed_log_file
    model_with_Path = network_path+"/"+model_name+".onnx"

    torch.manual_seed(0)
    # choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## generate 10 random input images based on the provided dimensions
    #generate_random_images(gen_img_input_dim_w, gen_img_input_dim_h, channels=3, count=20, path=quant_image_path)
    generate_random_images(imgDimW, imgDimH, imgChannel, count=20, path=quant_image_path)

    [pArray,status] = auto_profile(model,
                            loop,
                            imgChannel,
                            imgDimH,
                            imgDimW,
                            model_with_Path,
                            test_input_data,
                            perform_script_abs,
                            parse_script_abs,
                            perform_log_file_abs,
                            parsed_log_file_abs,
                            debug)

    print("------------- auto_profile done!...")

    return pArray,status
