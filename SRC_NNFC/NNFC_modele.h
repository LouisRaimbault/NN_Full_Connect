#ifndef NNFC_MODELE_H
#define NNFC_MODELE_H

#include "General.h"
#include "NNFC_init_modele.h"


float NNFC_Linear_Combine (float * Tab_val, int nb_in, int num_out);
void NNFC_Forward (NN_Full_Connect * NNFC);
void NNFC_do_Backward (Layer * L);
void NNFC_Backward (NN_Full_Connect * NNFC);
void NNFC_Train_Modele (Data * D);
void NNFC_Set_Yhat (Data *D, int t_o_t);
void NNFC_init_Coeff (Data * D);
void NNFC_Grad_updt (Grad * G, int nb_ind ,int a);
void NNFC_Modele (Data * D);


#endif