#ifndef ORDER_H
#define ORDER_H

#include "General.h"



void Cfer_tri_groupe (int * sumvec, int * tab_help,  int nb );
void order_scale_1 (int * tab_help, int * tab_val, int nb, int min, int max);
void Cfer_r_order (int * tr_sumvec, int * tab_help, int * tr_help, int iinf,int isup,int nb );
void Cfer_order(int * tab_help, int * sumvec ,int nb_elem, int maxul);
void special_float_order (float * tab_to_order, int nb_elem);

#endif