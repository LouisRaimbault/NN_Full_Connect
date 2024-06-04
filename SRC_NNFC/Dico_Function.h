#ifndef DICO_FUNCTION_H
#define DICO_FUNCTION_H

#include "General.h"
#include "Order.h"

float Activ_None (float val);
float deri_Activ_None (float val);
float Activ_Relu (float val);
float deri_Activ_Relu (float val);
float Activ_Sigmoide (float val);
float deri_Activ_Sigmoide (float val);
float Cost_MSE (float * val, int nb_arg);
float deri_Cost_MSE (float * val, int nb_arg);
float Cost_Cross_E (float * val, int nb_arg);
float deri_Cost_Cross_E (float * val, int nb_arg);
void Qual_RegLin (float * val) ;
void Qual_Classif (float * val);

float updt_LR_None (float * val);
void Set_map_Activ (std::unordered_map<std::string,float(*)(float)> * map_Activ);
void Set_map_deri_Activ (std::unordered_map<std::string,float(*)(float)> * map_deri_Activ);
void Set_map_Cost (std::unordered_map<std::string,float(*)(float *,int)> * map_Cost);
void Set_map_deri_Cost (std::unordered_map<std::string,float(*)(float *,int)> * map_deri_Cost);
void Set_map_Quality_Coeff (std::unordered_map<std::string,void(*)(float *)> * map_Quality_Coeff);
void Set_map_updt_LR (std::unordered_map<std::string,float(*)(float *)> * map_updt_LR);
void Set_Mappy_Star (Data * D);

#endif