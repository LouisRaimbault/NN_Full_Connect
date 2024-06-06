#ifndef APPLY_MODELE_H
#define APPLY_MODELE_H

#include "Apply_Modele_init.h"
#include "General.h"
#include "NNFC_modele.h"

void Apply_Modele_set_NNFC (Data * D);

void Apply_Modele_export_res(Data * D, float * Y_hat);

void Apply_Modele_Tsk (Data *D);

void Apply_Modele (Data * D);

#endif