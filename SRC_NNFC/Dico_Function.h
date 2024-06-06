#ifndef DICO_FUNCTION_H
#define DICO_FUNCTION_H

#include "General.h"
#include "Order.h"

float Activ_None (float val);
float deri_Activ_None (float val);
float Activ_Relu (float val);
float deri_Activ_Relu (float val);
float Activ_Leaky_Relu (float val);
float deri_Activ_Leaky_Relu (float val);
float Activ_Sigmoide (float val);
float deri_Activ_Sigmoide (float val);
float Activ_Tanh (float val);
float deri_Activ_Tanh(float val);

float Cost_MSE (float * val, int nb_arg);
void deri_Cost_MSE (float * val, float * deri_val ,int nb_arg);
float Cost_Cross_E (float * val, int nb_arg);
void deri_Cost_Cross_E (float * Val_Fwd, float * deri_val ,int nb_arg);

void Qual_Pred (float * Y, float * Y_hat, int nb);
void Qual_Classif (float * Y, float * Y_hat, int nb);

void updt_LR_None (Grad * G);
void updt_LR_Adagrad (Grad * G);
void updt_LR_Rmsprop (Grad * G);
void updt_LR_Adam (Grad * G);

void Set_map_Activ (std::unordered_map<std::string,float(*)(float)> * map_Activ);
void Set_map_deri_Activ (std::unordered_map<std::string,float(*)(float)> * map_deri_Activ);
void Set_map_Cost (std::unordered_map<std::string,float(*)(float *, int)> * map_Cost);
void Set_map_deri_Cost (std::unordered_map<std::string,void(*)(float *, float * ,int)> * map_deri_Cost);
void Set_map_Quality_Coeff (std::unordered_map<std::string,void(*)(float *, float*, int)> * map_Quality_Coeff);
void Set_map_updt_LR (std::unordered_map<std::string,void(*)(Grad *)> * map_updt_LR);
void Set_map_Alpha_Beta (std::unordered_map<std::string, int> * map_Alpha_Beta);

void Set_Mappy_Star (Data * D, int t);

#endif