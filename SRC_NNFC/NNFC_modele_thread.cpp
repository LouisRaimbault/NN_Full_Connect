#include "NNFC_modele_thread.h"

std::mutex mtx;
std::condition_variable cv;
int nbt = 0;
int nb_thread = 0;


void NNFC_Copie_NNFC (NN_Full_Connect * Master, NN_Full_Connect * Clone)
{
	Clone->F_Cost = Master->F_Cost;
	Clone->dF_Cost = Master->dF_Cost;

	Clone->nb_Layer = Master->nb_Layer;
	Clone->Tab_Layer = (Layer**)malloc(Clone->nb_Layer*sizeof(Layer*));
	for (int i = 0; i < Clone->nb_Layer; i++) {Clone->Tab_Layer[i] = (Layer*)malloc(sizeof(Layer));}

	Layer * LC = Clone->Tab_Layer[0];
	Layer * LM = Master->Tab_Layer[0];
	LC->nb_in = LM->nb_in;
	LC->nb_out = LM->nb_out;
	LC->Poids = LM->Poids;
	LC->dPoids = (float*)malloc(LC->nb_in*LC->nb_out*sizeof(float));
	LC->Z = (float*)malloc(LC->nb_out*sizeof(float));
	LC->F_Activ = LM->F_Activ;
	LC->Son = Clone->Tab_Layer[1];

	for (int i = 1; i < Clone->nb_Layer-1; i++)
		{
			LC = Clone->Tab_Layer[i];
			LM = Master->Tab_Layer[i];
			LC->nb_in = LM->nb_in;
			LC->nb_out = LM->nb_out;
			LC->Input = (float*)malloc(LC->nb_in*LC->nb_out*sizeof(float));
			LC->dInput = (float*)malloc(LC->nb_in*LC->nb_out*sizeof(float));
			LC->Poids = LM->Poids;
			LC->dPoids = (float*)malloc(LC->nb_in*LC->nb_out*sizeof(float));
			LC->Z = (float*)malloc(LC->nb_out*sizeof(float));
			LC->F_Activ = LM->F_Activ;
			LC->Father = Clone->Tab_Layer[i-1];
			LC->dF_Activ = LM->dF_Activ;
			LC->Alpha_Beta = LC->Father->Z;
			if (LM->Alpha_Beta == LM->Input) {LC->Alpha_Beta = LC->Input;}
			LC->Son = Clone->Tab_Layer[i+1];
		}

	LC = Clone->Tab_Layer[Clone->nb_Layer-1];
	LM = Master->Tab_Layer[Master->nb_Layer-1];
	LC->nb_in = LM->nb_out;
	LC->nb_out = LM->nb_out;
	LC->Input = (float*)malloc((LC->nb_in+1)*sizeof(float));// +1 car on rajoutera la valeur cur_Y
	LC->dInput = (float*)malloc(LC->nb_in*sizeof(float));
	LC->Father = Clone->Tab_Layer[Clone->nb_Layer-2];
	LC->Alpha_Beta = LC->Father->Z;
	if (LM->Alpha_Beta == LM->Input) {LC->Alpha_Beta = LC->Input;}

	Clone->nb_val_Fwd = LC->nb_out;
	Clone->Val_Fwd = LC->Input;
}


void NNFC_Copie_Grad (Grad * Master, Grad * Clone, NN_Full_Connect * NNFClone)
{
	Clone->nb_coeff = Master->nb_coeff;
	Clone->pt_dCoeff = (float**)malloc(Clone->nb_coeff*sizeof(float*));
	Clone->dCoeff_updt = (float*)malloc(Clone->nb_coeff*sizeof(float));
	int nb_coeff = 0;
	for (int i = 0; i < NNFClone->nb_Layer-1; i++)
		{	
			Layer * L = NNFClone->Tab_Layer[i];
			for (int j = 0; j < L->nb_in*L->nb_out; j++)
				{
					Clone->pt_dCoeff[nb_coeff] = &(L->dPoids[j]);
					nb_coeff = nb_coeff+1;
				}
		}
	NNFClone->G = Clone;
}

void Update_Grad_Master (Grad ** Tab_Grad, int nb_ind)
{
	Grad * Master = Tab_Grad[0];
	float * dC_Master = Master->dCoeff_updt;

	for (int i = 1; i < nb_thread; i++)
		{
			float * dC = Tab_Grad[i]->dCoeff_updt;
			for (int j = 0 ; j < Master->nb_coeff; j++)
				{
					dC_Master[j] = dC_Master[j] + dC[j];
				}
		}

	for (int j = 0; j < Master->nb_coeff; j++) {dC_Master[j] = dC_Master[j]/(float)nb_ind;}
}


void NNFC_Set_Info_Thread (Data * D)
{
	D->nb_thread = D->NC->nb_thread;
	NN_Full_Connect ** Th_NNFC = (NN_Full_Connect **)malloc(D->nb_thread*sizeof(NN_Full_Connect*));
	Grad ** Th_Grad = (Grad**)malloc(D->nb_thread*sizeof(Grad*));
	Th_NNFC[0] = D->NNFC;
	Th_Grad[0] = D->NNFC->G;
	for (int i = 1; i < D->nb_thread; i++)
		{
			Th_NNFC[i] = (NN_Full_Connect*)malloc(sizeof(NN_Full_Connect));
			NNFC_Copie_NNFC (Th_NNFC[0],Th_NNFC[i]);
			Th_Grad[i] = (Grad*)malloc(sizeof(Grad));
			NNFC_Copie_Grad(Th_Grad[0],Th_Grad[i],Th_NNFC[i]);
		}
	int ** Th_ind = (int**)malloc(D->nb_thread*sizeof(int*));
	int * Th_nb_ind = (int*)malloc(D->nb_thread*sizeof(int));

	int nb = D->Train->nb_ind/D->nb_thread;
	int w = D->Train->nb_ind%D->nb_thread;

	int sit = 0;
	for (int i = 0; i < w; i++)
		{
			Th_nb_ind[i] = nb+1;
			Th_ind[i] = (int*)malloc(Th_nb_ind[i]*sizeof(int));
			for (int j = 0; j < Th_nb_ind[i]; j++) {Th_ind[i][j] = sit; sit++;}
		}
	for (int i = w; i < D->nb_thread;i++ )
		{
			Th_nb_ind[i] = nb;
			Th_ind[i] = (int*)malloc(Th_nb_ind[i]*sizeof(int));
			for (int j = 0; j < Th_nb_ind[i]; j++) {Th_ind[i][j] = sit; sit++;}
		}

	if (sit != D->Train->nb_ind) {std::cout << "sit = " << sit << "alors que nb_ind = " << D->Train->nb_ind << "\n"; exit(1);}
	Info_Thread * ITh = (Info_Thread*)malloc(sizeof(Info_Thread));
	ITh->Th_NNFC = Th_NNFC;
	ITh->Th_Grad = Th_Grad;
	ITh->Th_ind = Th_ind;
	ITh->Th_nb_ind = Th_nb_ind;

	D->ITh = ITh;

}


void NNFC_Train_Modele_Thread (NN_Full_Connect * NNFC, Grad * G, int * ind, int nb_ind, Data * D)
{
	int nb_epoch = D->NNFC->nb_epoch;
	float ** X = D->Train->X;
	float * Y = D->Train->Y;
	float ** Input_0 = &(NNFC->Tab_Layer[0]->Input);

	for (int i = 0; i < nb_epoch; i++)
		{
			NNFC_Grad_updt(G,nb_ind,0);
			for (int j = 0; j < nb_ind; j++)
				{
					NNFC->cur_Y = Y[ind[j]];
					*(Input_0) = X[ind[j]];
					NNFC_Forward(NNFC);
					NNFC_Backward(NNFC);
					NNFC_Grad_updt(G,nb_ind,1);
				}

			std::unique_lock<std::mutex> lock(mtx);
			nbt = nbt+1;
			if (nbt == nb_thread)
				{
					nbt = 0;
					Update_Grad_Master (D->ITh->Th_Grad, D->nb_ind);
			        D->NNFC->F_updt_lr(D->NNFC->G);				
					cv.notify_all();
				}
			else 
				{
				   cv.wait(lock,[] {return nbt == 0;});
				}	
		}
}

void NNFC_Modele_Thread (Data * D)
{
	NNFC_init_modele (D);
	NNFC_init_Coeff (D);
	//
	//D->nb_thread = 1;
	//nb_thread = 1;
	//D->NC->nb_thread =1;

	//
	NNFC_Set_Info_Thread(D);
	nb_thread = D->nb_thread;

	std::thread tab_thread [nb_thread];
	Info_Thread * ITh = D->ITh;
	for (int i = 0; i < nb_thread; i++)
		{  
		   tab_thread[i] = std::thread(NNFC_Train_Modele_Thread,ITh->Th_NNFC[i],ITh->Th_Grad[i],ITh->Th_ind[i],ITh->Th_nb_ind[i],D);
		}
	for (int i = 0; i < nb_thread; i++) {tab_thread[i].join();}	
	NNFC_Set_Yhat (D,0);
	NNFC_Set_Yhat (D,1);
}


