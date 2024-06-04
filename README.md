# NN_FULL_Connect

## Description 

This page aims to propose different implementations in C/C++ (CPP extension) for fully connected neural networks.
The proposed algorithms do not use existing libraries for neural networks. This is my implementation of calculation graphs. Weight optimization is carried out by automated differentiation.

These implementations aim to be simple to use, sending only a configuration file and the database as call parameters to the program.

This version v1 can be used for classification or prediction, using a small number of activation functions available.
Multi-threaded optimization is available.

A future update (soon) will allow the final network to be extracted to reuse it with bases where Predictions/Classifications can be done, without training.

Version v2 will be a vectorized version. Then, the convolution or recurrence methods will be added.

It is important to know that the precision used in all algorithms is float precision.



**SRC_NNFC:**
* Model training
* Configuration file templates 
* List of activation functions



## Creating binaries and getting started
```
cd SRC_NNFC && make
```

## Dataset format 

For this version. The accepted data format is as follows. The first line is the name of the variables. Variables can be quantitative or qualitative. Binary or ordinal data are considered quantitative.

|v1,v2,v3|
|------------|
|1,a,2.21|
|2,b,3.07|
|-1,c,1.00|
|0,a,2.27|


The folder sample contain a simple small dataset test. You can use it for the example below to see how the software works.


### Example
```
SufRec in src_Fim

./NNFC ../Base_Test/Base.txt Config_Pred.txt  \\ Do prediction on Base.txt
./NNFC ../Base_Test/Mushroom.txt Config_Class.txt \\ Do classification on Mushroom.txt

```




