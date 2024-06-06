# NNFC

## Description 

**All parameters and hyperparameters must be specified in a configuration file.**

2 templates are available in this folder. Each line must end with a comma. If several parameters are to be entered on the same line, the separator is also a comma. The help.cpp file contains the list of parameters.\
If no value is to be specified, use "none". A correction is applied to the number of neurons in the input and output layers. So that it is consistent.\
 The training and test observations are selected randomly. If you want to carry out training on the entire database, set the value 0.00 in Ind_for_Test. A single observation will then be kept for the test base.\

Concerning configuration files. You must maintain the order of the lines and not delete them.



## Parameters for NNFC in SRC_NNFC (training Modele) :
|param|type|note|
|--------------------|--------|--------|
|    Target    |    string    | Name of The target |  
|    nb_Layer   |    int    | The number of Layers. Input and Output are considered as Layer | 
|    nb_Nodes_Layer  |    int [nb_Layer]    | The number of nodes/Layer | 
|    Range_first_teta  |    float[2]    | Borne inf and sup for first value of weigts   | 
|    Names_F_Activ   |    string[nb_layer-1] | Name of the Activation function for each layer (the output has none) | 
|    Name_F_Cost    |   string   |  Name of the Cost function    | 
|    Learning_Rate    |    float    | Value of the learning rate|
|    Name_F_Learate    |    string    | Name of the optimization function for the learning rate | 
|    Values_hyparam    |    float[2]   | Values for hyperparameters considering the Name_F_Learate| 
|    nb_epoch    |  int | number of epoch|
|    Name_F_Quality    |  string | Results display function|
|    Ind_for_Test    |  float | proportion of observations for Test Base|
|    nb_epoch    |  int | number of epoch|
|    Do Normalization    |  int | 1 if the data should be centered reduced (0 otherwise)|
|    Nb_Thread    |  int | number of threads to use|
|    Pathout    |  string | The path for output, usefull to apply Model. ("none" for no export)|


## Parameters for NNFC in SRC_NNFC (Use Modele and do Pred Classif) :

Most of parameters of this configuration file are already written during extraction

|param|type|note|
|--------------------|--------|--------|
|    Classif or Pred    |    string    | If model for Classif or Pred |  
|    Normalize or no   |    int    | If model was train with normalization of data | 
|    Nb_Var  |    int    | Number of explicativ variables (X) | 
|    Name X  |    string    | Names of explicativ variables (with order)  | 
|    avg_X   |    float[Nb_Var] | Average of explicativ variables (with order) | 
|    sd_X    |   float[Nb_Var]   |  Sd of explicativ variables (with order)   | 
|    Name_target    |    string    | Name of the target Y|
|    Nb_Mod_Y    |    int    | Number of modality for target Y| 
|    Mod_Y    |    string[Nb_Mod_Y]   | Names of Y mod| 
|    avg_Y    |  float | Average of Y|
|    sd_Y    |  float | sd of Y|
|    Nb_Layer    |  int | Number of Layers in Model|
|    Nb_Nodes_Layer    |  int[Nb_Layer] | Numer of nodes per Layer (cste not considered)|
|    Names_F_Activ   |    string[nb_layer-1] | Name of the Activation function for each layer (the output has none) |
|    nb_Weights    |  int | total number of Weights|
|    Weights    |  float[nb_Weights] | List of Weights (order is important)|
|    With_tgt    |  1 | If Target exist in new data (if 1 Test model, 0 do Class/Pred)|
|    pathout    |  string | if With_tgt = 0 , Get Pred/Class|



