#include "Apply_Modele.h"

void Apply_Modele_set_NNFC (Data * D)
{
	std::cout << "APPLY MODELE SET NNFC 0 \n";
	Modele_config * MC = D->MC;
	float * weights = MC->weights;
	int nb_weights = 0;
	NN_Full_Connect * NNFC = (NN_Full_Connect*)malloc(sizeof(NN_Full_Connect));
	std::cout << *(MC->COP) << "\n";
	NNFC->F_Quality = MC->MS->map_Quality_Coeff->operator[](*(MC->COP));
  std::cout << "APPLY MODELE SET NNFC 1 \n";
	NNFC->nb_Layer = MC->nb_Layer;
	NNFC->Tab_Layer = (Layer**)malloc((MC->nb_Layer)*sizeof(Layer*)); // +1 car ajout de la racine 
	for (int i = 0; i < MC->nb_Layer; i++) {NNFC->Tab_Layer[i] = (Layer*)malloc(sizeof(Layer));}
  std::cout << "APPLY MODELE SET NNFC 2 \n";
	Layer * L = NNFC->Tab_Layer[0];
	L->nb_in = MC->nb_Nodes_Layer[0]+1;
	if (D->MC->do_normalization == 1) {L->nb_in = MC->nb_Nodes_Layer[0];}
	L->nb_out = MC->nb_Nodes_Layer[1];
	L->Poids = (float*)malloc(L->nb_out*L->nb_in*sizeof(float));
	for (int j = 0; j < L->nb_out * L->nb_in; j++) {L->Poids[j] = weights[nb_weights]; nb_weights = nb_weights+1;}

	L->Z = (float*)malloc(L->nb_out*sizeof(float));
	L->F_Activ = MC->MS->map_Activ->operator[](MC->Name_F_Activ[0]);
	L->Son = NNFC->Tab_Layer[1];
	std::cout << "APPLY MODELE SET NNFC 3 \n";
	// End section Input Layer 
	for (int i = 1; i < MC->nb_Layer-1;i++)
		{
			L = NNFC->Tab_Layer[i];
			L->nb_in = MC->nb_Nodes_Layer[i]+1; // +1 pour la Constante 
			L->nb_out = MC->nb_Nodes_Layer[i+1];
			L->Input = (float*)malloc(L->nb_in*sizeof(float));
			L->Poids = (float*)malloc(L->nb_in*L->nb_out*sizeof(float));
			for (int j = 0; j < L->nb_in * L->nb_out; j++) {L->Poids[j] = weights[nb_weights]; nb_weights = nb_weights+1;}
			L->Z = (float*)malloc(L->nb_out*sizeof(float));
			L->F_Activ = MC->MS->map_Activ->operator[](MC->Name_F_Activ[i]);
			L->Father = NNFC->Tab_Layer[i-1];
			L->Son = NNFC->Tab_Layer[i+1];
		}	
	// Layer out Ayant de in que de out car on fait seulement la combinaison linéaire 
  std::cout << "APPLY MODELE SET NNFC 4 \n";
	L = NNFC->Tab_Layer[NNFC->nb_Layer-1];
	L->nb_in = D->nb_mod_Y;
	L->nb_out = D->nb_mod_Y;
	L->Input = (float*)malloc((L->nb_in+1)*sizeof(float));// +1 car on rajoutera la valeur cur_Y
	L->Father = NNFC->Tab_Layer[NNFC->nb_Layer-2];
	// End Layer Out
	// Partie Gradiant 
	if (nb_weights != MC->nb_weights) {std::cout << "Probleme in Coeff \n"; exit(1);}
	NNFC->nb_val_Fwd = L->nb_out;
	NNFC->Val_Fwd = L->Input;
	D->NNFC = NNFC;
}


void Apply_Modele_export_res(Data * D, float * Y_hat)
{
  std::ofstream outfile;

  outfile.open(*(D->MC->pathout));
  int i;
  if (outfile.is_open() == 0)
    {
      std::cerr << "Problem with output file \n";
      return ;
    }

  outfile << *(D->NomTarget);
  for (int i = 0; i < D->nb_ind; i++)
  	{
  		outfile << "\n" << Y_hat[i];
  	}

  outfile.close();
}

void Apply_Modele_Tsk (Data *D)
{
  NN_Full_Connect * NNFC = D->NNFC;
  std::string nom_base = "Base using Modele";
  int nb_ind = D->nb_ind;
  int nb_var = D->nb_var;
  float * Y = D->Y;
  float * Y_hat = (float*)malloc(nb_ind*sizeof(float));
  if (D->nb_mod_Y == 1) // Prediction
  {
  	for (int i = 0; i < nb_ind; i++)
		{	
				NNFC->cur_Y = Y[i];
				NNFC->Tab_Layer[0]->Input = D->X[i];
				NNFC_Forward(NNFC);
				Y_hat[i] = NNFC->Val_Fwd[0];
		}

		if (D->NC->do_normalization == 1)
			{
				for (int i =0; i < nb_ind;i++)
					{
						Y_hat[i] = Y_hat[i]*D->sd_Y + D->avg_Y;
					}
			}
	}

	if (D->nb_mod_Y > 1) // CLassification
  {
  	std::cout << "Il y a " << D->nb_mod_Y << " modalites sur la cible \n";
  	Y[nb_ind] = (float)D->nb_mod_Y;
  	int max = 0; float probmax;
  	for (int i = 0; i < D->nb_ind; i++)
		{	
				NNFC->cur_Y = Y[i];
				NNFC->Tab_Layer[0]->Input = D->X[i];
				NNFC_Forward(NNFC);
				max = 0; probmax = NNFC->Val_Fwd[0];
				for (int j = 1; j < D->nb_mod_Y; j++) {if (NNFC->Val_Fwd[j] > probmax) {probmax = NNFC->Val_Fwd[j]; max = j;}}
				Y_hat[i] = (float)max;
		}
	}

  std::cout << "results on Base : " << nom_base << "\n\n";
  if (D->MC->with_tgt == 1) {D->NNFC->F_Quality (Y,Y_hat,nb_ind);}
  if (D->MC->with_tgt == 0) {Apply_Modele_export_res(D,Y_hat);}
  free(Y_hat);

}



void Apply_Modele (Data * D)
{
	Apply_Modele_set_NNFC (D);
	Apply_Modele_Tsk (D);
}