#ifndef APPLY_MODELE_INIT_H
#define APPLY_MODELE_INIT_H
#include "General.h"


void Apply_Modele_DEL_MC (Modele_config * MC);
void Apply_Modele_Del_NNFC (NN_Full_Connect * NNFC);
void Apply_Modele_clean_mem (Data * D);
void Apply_Modele_Print_Config( Modele_config * MC);
void Apply_Modele_Normalization (Data * D, Modele_config * MC);
void Apply_Modele_Get_Modele (Modele_config * MC);
void Apply_Modele_Get_Data (Data * D, char * pathdata, char sep);

#endif