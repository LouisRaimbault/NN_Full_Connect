#include "Apply_Modele_init.h"

void Apply_Modele_DEL_MC (Modele_config * MC)
{
	delete MC->COP;
	delete MC->NomTarget;

	delete [] MC->Nom_X;
	delete [] MC->Name_F_Activ;
	delete [] MC->nomModY;
	delete MC->pathout;

	free(MC->nb_Nodes_Layer);
	free(MC->weights);
	free(MC->avg_X);
	free(MC->sd_X);

  NNFC_DEL_Mappy_Star (MC->MS);
	free(MC);
}

void Apply_Modele_Del_NNFC (NN_Full_Connect * NNFC)
{
	free(NNFC->Tab_Layer[0]->Z);
	free(NNFC->Tab_Layer[0]->Poids);
	free(NNFC->Tab_Layer[0]);

	for (int i = 1; i < NNFC->nb_Layer-1; i++)
		{
			free(NNFC->Tab_Layer[i]->Input);
			free(NNFC->Tab_Layer[i]->Z);
			free(NNFC->Tab_Layer[i]->Poids);
			free(NNFC->Tab_Layer[i]);
		}

	free(NNFC->Tab_Layer[NNFC->nb_Layer-1]->Input);
	free(NNFC->Tab_Layer[NNFC->nb_Layer-1]);


	free(NNFC->Tab_Layer);	
	free(NNFC);	
}

void Apply_Modele_clean_mem (Data * D)
{

	Apply_Modele_DEL_MC(D->MC);
	Apply_Modele_Del_NNFC(D->NNFC);
	free(D->Y);
  for (int i = 0; i < D->nb_ind; i++) {free(D->X[i]);} free(D->X);
	free(D);
}

void Apply_Modele_Print_Config( Modele_config * MC)
{

  std::cout << "################################################### \n\n";
  std::cout << "Caracteristiques du Modele entrainé \n";
  std::cout << "Target : " << *(MC->NomTarget) << "\n";
  std::cout << "Nb Layers : " << MC->nb_Layer << "\n";
  std::cout << "Nb Ner/Layer : ";
  for (int i = 0; i < MC->nb_Layer; i++) {std::cout << "[" << MC->nb_Nodes_Layer[i] << "] ";} std::cout << "\n";
  std::cout << "Activation Functions/Layers : ";
  for (int i = 0; i < MC->nb_Layer; i++) {std::cout << "[" << MC->Name_F_Activ[i] << "] ";} std::cout << "\n";
  std::cout << "Type : " << *(MC->COP) << "\n"; 
  std::cout << "Do Normalization : " << MC->do_normalization << "\n"; 
  std::cout << "Nb Weights : " << MC->nb_weights << "\n";
  std::cout << "pathout : " << *(MC->pathout) << "\n";
  std::cout << "\n ################################################### \n\n";

}

void Apply_Modele_Normalization (Data * D, Modele_config * MC)
{
	std::cout << "Apply Modele Normalization 0 \n";
	float * avg_X = MC->avg_X;
	float * sd_X = MC->sd_X;
	float * pt;

	for (int i = 0; i < D->nb_ind; i++)
		{
			pt = D->X[i];
			for (int j = 0; j < D->nb_var; j++)
				{
					pt[j] = (pt[j]- avg_X[j]) / sd_X[j];
				}
		}
}

void Apply_Modele_Get_Modele (Modele_config * MC)
{
	uint64_t l;
	int i,j,nrow;
	char * Buffer;
	fill_arr_from_file(MC->Path_Config,&l,&nrow,&Buffer);
	i = 0; std::string st = "";
	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Info : Classif Or Pred
	MC->COP = new std::string (st); st = "";

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Info : do normalization
	MC->do_normalization = std::stoi(st); st = "";

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Info : nb_var
	MC->nb_var = std::stoi(st); st = "";


	MC->Nom_X = new std::string [MC->nb_var]; // Nom des variables (important ordre)
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_var; j++)
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->Nom_X[j] = st; st = "";
		}

	if (MC->do_normalization == 0)
		{
			while (Buffer[i] != ',') {i++;} i++;
			while (Buffer[i] != ',') {i++;} i++;
			goto skip_norm;
		}	

	MC->avg_X = (float*)malloc(MC->nb_var*sizeof(float)); // Moyenne des Xi
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_var; j++)
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->avg_X[j] = std::stof(st); st = "";
		}	

  	MC->sd_X = (float*)malloc(MC->nb_var*sizeof(float)); // sd des Xi
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_var; j++)
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->sd_X[j] = std::stof(st); st = "";
		}

	skip_norm : ;

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Nom Target
	MC->NomTarget = new std::string(st); st = "";


	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Nb Mod Y
	MC->nb_mod_Y = std::stoi(st); st = "";

	MC->nomModY = new std::string [MC->nb_mod_Y]; // Nom des mod y
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_mod_Y; j++)
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->nomModY[j] = st; st = "";
		}

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Avg Y
	MC->avg_Y = std::stof(st); st = "";

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // sd Y
	MC->sd_Y = std::stof(st); st = "";

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Nb Layer
	MC->nb_Layer = std::stoi(st); st = "";

	MC->nb_Nodes_Layer = (int*)malloc(MC->nb_Layer*sizeof(int)); // Nombre de neouds par layer
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_Layer; j++)
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->nb_Nodes_Layer[j] = std::stoi(st); st = "";
		}

	MC->Name_F_Activ = new std::string [MC->nb_Layer];
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_Layer; j++) // Nom des fonctions d'activations
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->Name_F_Activ[j] = (st); st = "";
		}

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Nb Total de poids
	MC->nb_weights = std::stoi(st); st = "";

	MC->weights = (float*)malloc(MC->nb_weights*sizeof(float));
	while (Buffer[i] != ':') {i++;} i++;
	for (j = 0; j < MC->nb_weights; j++) // Récupération des poids
		{
			while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
			MC->weights[j] = std::stof(st); st = "";
		}
	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // File avec target
	MC->with_tgt = std::stoi(st); st = "";

	while (Buffer[i] != ':') {i++;} i++;
	while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // File avec target
	MC->pathout = new std::string(st); st = "";
 
 
  free (Buffer);
  Apply_Modele_Print_Config(MC);
}

void Apply_Modele_Get_Data (Data * D, char * pathdata, char sep)
{
	int i,j;
	float f1;
	float * ptf;
	NNFC_Get_Data (D, pathdata, sep);
	if (D->MC->nb_mod_Y != D->nb_mod_Y) {std::cout << "Problem of possible outputs \n"; exit(1);}

	Modele_config * MC = D->MC;
	int nb_var = MC->nb_var;
	int nb_obs = D->nb_ind;
	float ** X = (float**)malloc((nb_obs)*sizeof(float*));
	float * Y = (float*)malloc((nb_obs+5)*sizeof(float));
	std::unordered_map<std::string,int> mappX;
	for (i = 0; i < MC->nb_var; i++) {mappX[MC->Nom_X[i]] = i+1;}
	int * corresponding_var = (int*)malloc(MC->nb_var*sizeof(int));
	for (i = 0; i < D->nb_var; i++)
		{
			if (mappX.count(D->Nomvar[i]) == 0) 
				{corresponding_var[i] = D->nb_var; std::cout << "New variables, non considered \n"; continue;}
			corresponding_var[i] = mappX[D->Nomvar[i]]-1;
		}

	for (i = 0; i < nb_obs; i++)
		{
			X[i] = (float*)malloc((nb_var+1)*sizeof(float));
			ptf = X[i];
			for (j = 0; j < nb_var; j++)
				{
					ptf[corresponding_var[j]] = D->X[i][j];
				}
			X[i][nb_var] = 1.0f;
			free(D->X[i]);
		}
	free(D->X);

	if (MC->with_tgt == 0) 
		{
			for (i = 0; i < nb_obs; i++) {Y[i] = 0.0f;}
		}
	if (*(MC->COP) == "Pred" && MC->with_tgt == 1) 
		{
			for (i = 0; i < nb_obs; i++) {Y[i] = D->Y[i];}
		}
	if (*(MC->COP) == "Classif" && MC->with_tgt == 1) // s'assurer que l'ordre des Y correspond
		{
			std::unordered_map<std::string,int> mappFakeY; // Ordre sur nouv base
			std::unordered_map<std::string,int> mappyrealY; // Vrai ordre
			int * Tab_Change_Y = (int*)malloc(MC->nb_mod_Y*sizeof(int));
			for (i = 0; i < MC->nb_mod_Y; i++)
				{
					mappFakeY[D->nomModY[i]] = i;
					mappyrealY[MC->nomModY[i]] = i;
				}
			for (i = 0; i < MC->nb_mod_Y; i++)
				{
					Tab_Change_Y[mappFakeY[D->nomModY[i]]] = mappyrealY[D->nomModY[i]];
				}
			for (j = 0; j < nb_obs; j++) 
				{
					Y[j] = (float)Tab_Change_Y[(int)(D->Y[j]+0.01f)];
				}	
			free(Tab_Change_Y);
		}

	free(D->Y);
	delete [] D->nomModY;
	delete [] D->Nomvar;
	D->X = X;
	D->Y = Y;
	D->nb_mod_Y = MC->nb_mod_Y;
	D->nb_var = MC->nb_var;
	D->nomModY = MC->nomModY;
	D->Nomvar = MC->Nom_X;
	D->avg_X = MC->avg_X;
	D->sd_X = MC->sd_X; 
	D->avg_Y = MC->avg_Y;
	D->sd_Y = MC->sd_Y;
	if (MC->do_normalization == 1) {Apply_Modele_Normalization(D,MC);}	
	free(corresponding_var);
}