#include "NNFC_init_modele.h"

void shuffle(int * array, int n) 
{
	srand(time(0));
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void Normalize_Matrix (Data * D)
{
	int nb_ind = D->nb_ind;
	int nb_var = D->nb_var;
	int i,j;
	float * avg_X = (float*)malloc(nb_var*sizeof(float));
	float * avg_2_X = (float*)malloc(nb_var*sizeof(float));
	float * sd_X = (float*)malloc(nb_var*sizeof(float));
	float * ptf;
	for (j = 0; j < nb_var; j++) {avg_X[j] = 0.0f; avg_2_X[j] = 0.0f; sd_X[j] = 0.0f;}

	for (i = 0; i< nb_ind; i++)
		{
			ptf = D->X[i];
			for (j = 0; j < nb_var; j++)
				{
					avg_X[j] = avg_X[j] + ptf[j];
					avg_2_X[j] = avg_2_X[j] + ptf[j] * ptf[j];
				}
		}

	for (j = 0; j < nb_var; j++) 
		{
			avg_X[j] = avg_X[j]/(float)nb_ind;
			avg_2_X[j] = avg_2_X[j]/(float)nb_ind;
			sd_X[j] = avg_2_X[j] - avg_X[j]*avg_X[j];
			sd_X[j] = std::pow(sd_X[j],0.05f);
			if (sd_X[j] < 0.00001) {sd_X[j] = 0.00001;}
		}

	for (i = 0; i< nb_ind; i++)
		{
			ptf = D->X[i];
			for (j = 0; j < nb_var; j++)
				{
					ptf[j] = (ptf[j]-avg_X[j])/sd_X[j];
				}
		}	


	if (D->nb_mod_Y == 1)
		{
			float avg_Y = 0.0f;
			float avg_2_Y = 0.0f;
			float sd_Y = 0.0f;
			for (int i = 0; i < nb_ind; i++) 
				{
					avg_Y = avg_Y + D->Y[i];
					avg_2_Y = avg_2_Y + D->Y[i]*D->Y[i];
				}
			avg_Y = avg_Y/(float)nb_ind;
			avg_2_Y = avg_2_Y/(float)nb_ind;
			sd_Y = avg_2_Y - avg_Y*avg_Y;
			sd_Y = std::pow(sd_Y,0.05f);

			for (int i = 0; i < nb_ind; i++) {D->Y[i] = (D->Y[i]-avg_Y)/sd_Y;}
			D->avg_Y = avg_Y;
			D->sd_Y = sd_Y;
		}	
	D->avg_X = avg_X;	
	D->sd_X = sd_X;
	free(avg_2_X);

}

void NNFC_Set_Bases (Data * D)
{	
	int nb_ind = D->nb_ind;
	int nb_ind_test = (int)(D->nb_ind*D->NC->pcent_Test);
	int nb_ind_train = D->nb_ind - nb_ind_test;
	if (nb_ind_test == 0) {nb_ind_test = 1;}	
	float ** Mat_Train = (float**)malloc(nb_ind_train*sizeof(float*));
	float ** Mat_Test = (float**)malloc(nb_ind_test*sizeof(float*)); 
	int * ind_order = (int*)malloc(nb_ind*sizeof(int));
	for (int i = 0; i < nb_ind; i++) {ind_order[i] = i;}
	shuffle(ind_order,nb_ind);
	float * Y_Train = (float*)malloc((nb_ind_train+5)*sizeof(float)); // +5 pour param fonction print
	float * Y_Test = (float*)malloc((nb_ind_test+5)*sizeof(float)); // +5 pour param fonction print

	for (int i = 0; i < nb_ind_train; i++) 
		{
			Y_Train[i] = D->Y[ind_order[i]];
			Mat_Train[i] = D->X[ind_order[i]];
		}
	int j = nb_ind_train;
	for (int i = 0; i < nb_ind_test; i++) 
		{   
			Y_Test[i] = D->Y[ind_order[j]];
			Mat_Test[i] = D->X[ind_order[j]]; j++;
		}
	D->Train->nb_ind = nb_ind_train; D->Train->X = Mat_Train;
	D->Train->Y_hat = (float*)malloc(nb_ind_train*sizeof(float));
	D->Train->nb_var = D->nb_var;
	D->Train->Y = Y_Train;

	D->Test->nb_ind = nb_ind_test; D->Test->X = Mat_Test;
	D->Test->Y_hat = (float*)malloc(nb_ind_test*sizeof(float));
	D->Test->nb_var = D->nb_var;
	D->Test->Y = Y_Test;
	free (ind_order);
}

void NNFC_Init_Grad (Data * D, NN_Full_Connect * NNFC ,Grad ** pt_G)
{
	Grad * G = (Grad*)malloc(sizeof(Grad));
	G->lr = NNFC->learate;
	G->p1 = D->NC->Values_Hyparam[0]; 
	G->p2 = D->NC->Values_Hyparam[1]; 
	G->Epsilon = 0.000001;
	G->cur_epoch = 1.0f;
	Layer ** Tab_Layer = NNFC->Tab_Layer;
	int nb_coeff = 0;

	for (int i = 0; i < NNFC->nb_Layer-1; i++)
		{
			nb_coeff = nb_coeff + Tab_Layer[i]->nb_in * Tab_Layer[i]->nb_out;
		}
	G->pt_Coeff = (float**)malloc(nb_coeff*sizeof(float*));
	G->pt_dCoeff = (float**)malloc(nb_coeff*sizeof(float*));
	G->dCoeff_updt = (float*)malloc(nb_coeff*sizeof(float));
	G->G1 = (float*)malloc(nb_coeff*sizeof(float));
	G->G2 = (float*)malloc(nb_coeff*sizeof(float));
	for (int i = 0; i < nb_coeff; i++) {G->G1[i] = 0.00f; G->G2[i] = 0.00f;}
	nb_coeff = 0;
	for (int i = 0; i < NNFC->nb_Layer-1; i++)
		{	
			Layer * L = Tab_Layer[i];
			for (int j = 0; j < L->nb_in*L->nb_out; j++)
				{
					G->pt_Coeff[nb_coeff] = &(L->Poids[j]);
					G->pt_dCoeff[nb_coeff] = &(L->dPoids[j]);
					nb_coeff = nb_coeff+1;
				}
		}
	G->nb_coeff = nb_coeff;
	*(pt_G) = G;


}

void NNFC_Set_NNFC (Data * D, NN_Full_Connect ** pt_NNFC)
{
	NN_Full_Connect * NNFC = (NN_Full_Connect*)malloc(sizeof(NN_Full_Connect));
	NNFC_config * NC = D->NC;
	NC->nb_Nodes_Layer[0] = D->nb_var; // protection contre mauvais param 
	NC->nb_Nodes_Layer[NC->nb_Layer-1] = D->nb_mod_Y; // Protection contre mauvais param

	NNFC->F_Cost = NC->MS->map_Cost->operator[](*(NC->Name_F_Cost));
	NNFC->dF_Cost = NC->MS->map_deri_Cost->operator[](*(NC->Name_F_Cost));

	NNFC->F_Quality = NC->MS->map_Quality_Coeff->operator[](*(NC->Name_F_Quality));
	NNFC->F_updt_lr = NC->MS->map_updt_LR->operator[](*(NC->Name_F_Learate));

	NNFC->nb_Layer = NC->nb_Layer;
	NNFC->Tab_Layer = (Layer**)malloc((NC->nb_Layer)*sizeof(Layer*)); // +1 car ajout de la racine 
	for (int i = 0; i < NC->nb_Layer; i++) {NNFC->Tab_Layer[i] = (Layer*)malloc(sizeof(Layer));}
	//Layer Root
	NNFC->nb_epoch = NC->nb_epoch;
	NNFC->learate = NC->Learate;
	// Section input Layer
	Layer * L = NNFC->Tab_Layer[0];
	L->nb_in = NC->nb_Nodes_Layer[0]+1;
	if (D->NC->do_normalization == 1) {L->nb_in = NC->nb_Nodes_Layer[0];}
	L->nb_out = NC->nb_Nodes_Layer[1];
	L->Poids = (float*)malloc(L->nb_out*L->nb_in*sizeof(float));
	L->dPoids = (float*)malloc(L->nb_out*L->nb_in*sizeof(float));
	L->Z = (float*)malloc(L->nb_out*sizeof(float));
	L->F_Activ = NC->MS->map_Activ->operator[](NC->Name_F_Activ[0]);
	L->Son = NNFC->Tab_Layer[1];
	// End section Input Layer 
	for (int i = 1; i < NC->nb_Layer-1;i++)
		{
			L = NNFC->Tab_Layer[i];
			L->nb_in = NC->nb_Nodes_Layer[i]+1; // +1 pour la Constante 
			L->nb_out = NC->nb_Nodes_Layer[i+1];
			L->Input = (float*)malloc(L->nb_in*sizeof(float));
			L->dInput = (float*)malloc(L->nb_out*sizeof(float));
			L->Poids = (float*)malloc(L->nb_in*L->nb_out*sizeof(float));
			L->dPoids = (float*)malloc(L->nb_in*L->nb_out*sizeof(float));
			L->Z = (float*)malloc(L->nb_out*sizeof(float));
			L->F_Activ = NC->MS->map_Activ->operator[](NC->Name_F_Activ[i]);
			L->Father = NNFC->Tab_Layer[i-1];
			L->dF_Activ = NC->MS->map_deri_Activ->operator[](NC->Name_F_Activ[i-1]);
			L->Alpha_Beta = L->Father->Z;
			if (NC->MS->map_Alpha_Beta->operator[](NC->Name_F_Activ[i-1]) == 1) {L->Alpha_Beta = L->Input;}
			L->Son = NNFC->Tab_Layer[i+1];
		}	
	// Layer out Ayant de in que de out car on fait seulement la combinaison linéaire 

	L = NNFC->Tab_Layer[NNFC->nb_Layer-1];
	L->nb_in = D->nb_mod_Y;
	L->nb_out = D->nb_mod_Y;
	L->Input = (float*)malloc((L->nb_in+1)*sizeof(float));// +1 car on rajoutera la valeur cur_Y
	L->dInput = (float*)malloc(L->nb_in*sizeof(float));
	L->Father = NNFC->Tab_Layer[NNFC->nb_Layer-2];
	L->Alpha_Beta = L->Father->Z;
	if (NC->MS->map_Alpha_Beta->operator[](NC->Name_F_Activ[NC->nb_Layer-2]) == 1) {L->Alpha_Beta = L->Input;}
	// End Layer Out
	// Partie Gradiant 

	NNFC->nb_val_Fwd = L->nb_out;
	NNFC->Val_Fwd = L->Input;
	*(pt_NNFC) = NNFC;

}

void NNFC_init_modele (Data * D)
{
	D->NC->nb_Nodes_Layer[0] = D->nb_var;
	D->Train = (Base*)malloc(sizeof(Base)); D->Train->nb_ind = D->nb_ind;
	D->Test = (Base*)malloc(sizeof(Base)); D->Test->nb_ind = D->nb_ind;
	D->avg_Y = 0.00; D->sd_Y = 0.00;
	if (D->NC->do_normalization == 1) {Normalize_Matrix(D);}
	NNFC_Set_Bases(D);
	NNFC_Set_NNFC(D,&(D->NNFC));
	NNFC_Init_Grad(D,D->NNFC,&(D->NNFC->G));
}