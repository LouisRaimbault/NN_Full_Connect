#include "NNFC_modele.h"

float NNFC_Linear_Combine (float * Input, float * Poids, int nb_in, int num_out)
{
	float ret = 0.0f;
	int j = num_out*nb_in;
	for (int i = 0; i < nb_in; i++)
		{
			ret = ret + Input[i]*Poids[j]; j++;
		}
	return ret;
}


void NNFC_Forward (NN_Full_Connect * NNFC)
{
	int j; float ret;
	Layer * L;
	float * Son_Input;
	for (int i = 0; i < NNFC->nb_Layer-1; i++)
		{
			L = NNFC->Tab_Layer[i];
			Son_Input = L->Son->Input;
			for (j = 0; j < L->nb_out; j++)
				{ 
					L->Z[j] = NNFC_Linear_Combine(L->Input, L->Poids ,L->nb_in,j);
					Son_Input[j] = L->F_Activ(L->Z[j]); 
				}
			Son_Input[L->nb_out] = 1.0f; // Pour la valeur constante 
		}
}

void NNFC_do_Backward (Layer * L)
{
	float * Tab_Input = L->Input;
	float * Tab_Poids = L->Poids;
	int nb_in = L->nb_in;
	int nb_out = L->nb_out;
	float * dValS = L->Son->dInput;
	for (int i = 0; i < nb_in; i++) // On s'occupe des derivées de Z et des coeffs simultanément
		{
			float deriv = 0;
			for (int j = 0; j < nb_out; j++)
				{
					deriv = deriv + Tab_Poids[(nb_in*j)+i] * dValS[j]; // pour Z
					L->dPoids[(nb_in*j)+i] = Tab_Input[i] * dValS[j];
				}
			// Deriv est la derivée de la valeur f(Z), on veut la derivee de Z
			if (i != nb_in-1) {L->dInput[i] = L->dF_Activ(L->Alpha_Beta[i])*deriv; continue;} // Si autre que constante
			
		}
}

void NNFC_Backward (NN_Full_Connect * NNFC)
{	
	int i,j,z;
	int nb_Layer = NNFC->nb_Layer;
	Layer ** Tab_Layer = NNFC->Tab_Layer; 
	int nb_val = NNFC->nb_val_Fwd;
	int nb_arg = nb_val+1;
	NNFC->Val_Fwd[nb_val] = NNFC->cur_Y; // Inutil d'avoir ce cur Y mais bon je laisse temporairement

	NNFC->dF_Cost(NNFC->Val_Fwd,Tab_Layer[nb_Layer-1]->dInput,nb_arg);
	
	for (i = nb_Layer-2; i > 0; i--) // 
		{	
			NNFC_do_Backward (Tab_Layer[i]);
		}
	// Partie premire couche
	Layer * L = Tab_Layer[0];
	for (i = 0; i < L->nb_out; i++) // 
		{ z = i * L->nb_in;
			for (j = 0; j < L->nb_in; j++)
				{
					L->dPoids[z+j] = L->Input[j] * L->Son->dInput[i]; // A verifier 
				}
		}
}

void NNFC_Grad_updt (Grad * G, int nb_ind ,int a)
{   
	if (a == 1)
		{
			float * dC = G->dCoeff_updt;
			float ** ptd = G->pt_dCoeff;
			for (int i = 0; i < G->nb_coeff; i++) {dC[i] = dC[i] + *(ptd[i]);}
			return;
		}

	if (a == 2)
		{
			float fnb = (float)nb_ind;
			float * dC = G->dCoeff_updt;
			for (int i = 0; i < G->nb_coeff; i++) {dC[i] = dC[i]/fnb;}
			return;			
		}

	if (a == 0)
		{
			float * dC = G->dCoeff_updt;
			for (int i = 0; i < G->nb_coeff; i++) {dC[i] = 0.0f;}			
		}
}

void NNFC_Update_Coeff (Grad * G, float lr)
{
	for (int i = 0; i < G->nb_coeff; i++)
		{	
			*(G->pt_Coeff[i]) = *(G->pt_Coeff[i])- lr * G->dCoeff_updt[i];
			std::cout << *(G->pt_Coeff[i]) << "\n";
		}
}



void NNFC_Train_Modele (Data * D)
{
	NN_Full_Connect * NNFC = D->NNFC;
	Base * B = D->Train;
	Grad * G = NNFC->G;
	float * Y = B->Y;
	int i,j,k;
	int nb_epoch = NNFC->nb_epoch;
	float learate = NNFC->learate;
	int nb_ind = B->nb_ind;
	int nb_var = B->nb_var;

	for (i = 0; i < nb_epoch; i++) // nb_epoch
		{
			NNFC_Grad_updt (G,nb_ind,0);
			for (j = 0; j < nb_ind; j++) // nb_ind
				{	
					NNFC->cur_Y = Y[j];
					NNFC->Tab_Layer[0]->Input = B->X[j];
					NNFC_Forward(NNFC);
					NNFC_Backward(NNFC);
					NNFC_Grad_updt(G,nb_ind,1);
					
				}
			NNFC_Grad_updt(G,nb_ind,2);
			NNFC->F_updt_lr(NNFC->G);
		}
}

void NNFC_Set_Yhat (Data *D, int t_o_t)
{
  Base * B = D->Train;
  NN_Full_Connect * NNFC = D->NNFC;
  std::string nom_base = "Train";
  if (t_o_t == 1) {B = D->Test; nom_base = "Test";}
  float * Y = B->Y;
  float * Y_hat = B->Y_hat;
  int nb_ind = B->nb_ind;
  int nb_var = D->nb_var;
  if (D->nb_mod_Y == 1) // Prediction
  {
  	for (int i = 0; i < nb_ind; i++)
		{	
				NNFC->cur_Y = Y[i];
				NNFC->Tab_Layer[0]->Input = B->X[i];
				NNFC_Forward(NNFC);
				Y_hat[i] = NNFC->Val_Fwd[0];
		}

		if (D->NC->do_normalization == 1)
			{
				for (int i =0; i < nb_ind;i++)
					{
						Y_hat[i] = Y_hat[i]*D->sd_Y + D->avg_Y;
						Y[i] = Y[i]*D->sd_Y+ D->avg_Y;
					}
			}
	}

	if (D->nb_mod_Y > 1) // CLassification
  {
  	std::cout << "Il y a " << D->nb_mod_Y << " modalites sur la cible \n";
  	B->Y[nb_ind] = (float)D->nb_mod_Y;
  	int max = 0; float probmax;
  	for (int i = 0; i < nb_ind; i++)
		{	
				NNFC->cur_Y = Y[i];
				NNFC->Tab_Layer[0]->Input = B->X[i];
				NNFC_Forward(NNFC);
				max = 0; probmax = NNFC->Val_Fwd[0];
				for (int j = 1; j < D->nb_mod_Y; j++) {if (NNFC->Val_Fwd[j] > probmax) {probmax = NNFC->Val_Fwd[j]; max = j;}}
				Y_hat[i] = (float)max;
		}
	}

  std::cout << "results on Base : " << nom_base << "\n\n";
  D->NNFC->F_Quality (Y,Y_hat,nb_ind);
}


void NNFC_init_Coeff (Data * D)
{
	NN_Full_Connect * NNFC = D->NNFC;
	srand(time(NULL));
	float binf = D->NC->Range_First_Teta[0];
	float bsup = D->NC->Range_First_Teta[1];

	for (int i = 0; i < NNFC->nb_Layer-1; i++)
		{
			for (int j = 0; j < NNFC->Tab_Layer[i]->nb_in*NNFC->Tab_Layer[i]->nb_out; j++)
				{
					NNFC->Tab_Layer[i]->Poids[j] = binf + ((float)rand()/RAND_MAX)*(bsup-binf);
					//std::cout << "NNFC->Tab_Layer[i]->Poids[j] = " << NNFC->Tab_Layer[i]->Poids[j] << "\n";

				}
		}

}

void NNFC_Modele (Data * D)
{
	NNFC_init_modele (D);
	NNFC_init_Coeff (D);
	NNFC_Train_Modele (D);
	NNFC_Set_Yhat (D,0);
	NNFC_Set_Yhat (D,1);
}