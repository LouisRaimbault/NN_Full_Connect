#ifndef NNFC_INIT_MODELE
#define NNFC_INIT_MODELE

#include "General.h"


void shuffle(int * array, int n); 
void Normalize_Matrix (Data * D);
void NNFC_Set_Bases (Data * D);
void NNFC_Init_Grad (Data * D, NN_Full_Connect * NNFC ,Grad ** pt_G);
void NNFC_Set_NNFC (Data * D, NN_Full_Connect ** pt_NNFC);
void NNFC_init_modele (Data * D);

#endif