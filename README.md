# README #

AUTOMATION FLOW

### What is this repository for? ###

* implements the complete flow on ubuntu host for executing and profiling the converted model on khadas npu
* parses the results fetched from the khadas for timing informations

##   Content:
1. "nbg_unify" folder that contains the default makefiles abd build script for compiling a network
2. "normal_case" folder that contains the default makefiles abd build script for compiling a network for profiling
3. "perform.sh" performs the complete flow from host to khadas and get back the results
4. "parse.sh" parses the results fetched from the khadas for timing informations

##   Perform.sh:
#### Steps:

__Assumes that the conversion is already carried and converted files are already there at the host__

1. clone the repository on the host
2. you should have already added host ssh-key into remote khadas and vice-versa.
3. set the required variables in the script "perform.sh"; go through all the variables, the comments explain each of them.
4. check the path of the source (host) and destination(remote) in the script "perform.sh"
5. run the "perform.sh" from host. it copy required .h, .c, .nb, .export.data, to the khadas, compile the network on the khadas, execute there and profile there also.
6. the log for the executing network on the khadas and corresponding profiling will be logged in the "model_execution.log" and "model_profiling.log" on the host.

##   Perform.sh:
Parse the results fetched from the Khadas.
1. run the "parse.sh" on the results/logs taken from the execution/profiling on the khadas.
2. takes one input parameter: the file to be parsed; one output parameter: the parsed output file

