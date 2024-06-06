#include "General.h"
#include "NNFC_modele.h"
#include <thread>
#include <mutex>
#include <condition_variable>


void NNFC_Copie_NNFC (NN_Full_Connect * Master, NN_Full_Connect * Clone);
void NNFC_Copie_Grad (Grad * Master, Grad * Clone, NN_Full_Connect * NNFClone);
void Update_Grad_Master (Grad ** Tab_Grad, int nb_ind);
void NNFC_Set_Info_Thread (Data * D);
void NNFC_Train_Modele_Thread (NN_Full_Connect * NNFC, Grad * G, int * ind, int nb_ind, Data * D);
void NNFC_Modele_Thread (Data * D);


