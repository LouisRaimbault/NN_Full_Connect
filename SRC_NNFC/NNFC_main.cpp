#include "General.h"
#include "Dico_Function.h"
#include "NNFC_init_modele.h"
#include "NNFC_modele.h"
#include "NNFC_modele_thread.h"
#include "Apply_Modele_init.h"
#include "Apply_Modele.h"

int main (int argc , char ** argv )
{	
	char * path_data = argv[1];
	char * path_config = argv[2];
	int use_modele = std::atoi(argv[3]);

	if (use_modele == 0)
	{
		NNFC_config * NC = (NNFC_config*)malloc(sizeof(NNFC_config));
		NC->Path_Config = path_config;
		NNFC_Get_Config(NC);
		Data * D = (Data*)malloc(sizeof(Data));
		D->NC = NC;
		D->NomTarget = NC->NomTarget;
		NNFC_Get_Data (D, path_data, ',');
		Set_Mappy_Star (D,0);
		if (NC->nb_thread == 1) {NNFC_Modele(D);}
		if (NC->nb_thread > 1) {NNFC_Modele_Thread(D);}
		if (*(NC->pathout) != "none") {NNFC_Get_Export(D);}
		NNFC_DEL_Data (D);
	}

	if (use_modele == 1)
		{
			Modele_config * MC = (Modele_config*)malloc(sizeof(Modele_config));
			MC->Path_Config = path_config;
			Apply_Modele_Get_Modele (MC);
			Data * D = (Data*)malloc(sizeof(Data));
			D->MC = MC;
			D->NomTarget = MC->NomTarget;
			Apply_Modele_Get_Data (D,path_data,',');
			Set_Mappy_Star (D,1);
			Apply_Modele(D);
			Apply_Modele_clean_mem (D);

		}

	

    return 0;
}