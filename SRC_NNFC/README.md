# NNFC

## Description 

**All parameters and hyperparameters must be specified in a configuration file.**

2 templates are available in this folder. Each line must end with a comma. If several parameters are to be entered on the same line, the separator is also a comma. The help.cpp file contains the list of parameters. If no value is to be specified, use "none". A correction is applied to the number of neurons in the input and output layers. So that it is consistent. The training and test observations are selected randomly. If you want to carry out training on the entire database, set the value 0.00 in Ind_for_Test. A single observation will then be kept for the test base.

Concerning configuration files. You must maintain the order of the lines and not delete them.



## Parameters for SufRec in src_Fim :
|param|type|note|
|--------------------|--------|--------|
|    Target    |    string    | Name of The target |  
|    nb_Layer   |    int    | The number of Layers. Input and Output are considered as Layer | 
|    nb_Nodes_Layer  |    int [nb_Layer]    | The number of nodes/Layer | 
|    Range_first_teta  |    float * [2]    | Borne inf and sup for first value of weigts   | 
|    Names_F_Activ   |    string[nb_layer-1] | Name of the Activation function for each layer (the output has none) | 
|    Name_F_Cost    |   string   |  Name of the Cost function    | 
|    Learning_Rate    |    float    | Value of the learning rate|
|    Name_F_Learate    |    string    | Name of the optimization function for the learning rate | 
|    Values_hyparam    |    float[2]*   | Values for hyperparameters considering the Name_F_Learate| 
|    nb_epoch    |  int | number of epoch|
|    Name_F_Quality    |  string | Results display function|
|    Ind_for_Test    |  float | proportion of observations for Test Base|
|    nb_epoch    |  int | number of epoch|
|    Do Normalization    |  int | 1 if the data should be centered reduced (0 otherwise)|
|    Nb_Thread    |  int | number of threads to use|

