#include "General.h"
#include "Dico_Function.h"
#include "NNFC_init_modele.h"
#include "NNFC_modele.h"
#include "NNFC_modele_thread.h"

int main (int argc , char ** argv )
{	
	char * path_data = argv[1];
	char * path_config = argv[2];
	NNFC_config * NC = (NNFC_config*)malloc(sizeof(NNFC_config));
	NC->Path_Config = path_config;
	NNFC_Get_Config(NC);

	Data * D = (Data*)malloc(sizeof(Data));
	D->NC = NC;
	D->NomTarget = NC->NomTarget;
	NNFC_Get_Data (D, path_data, ',');
	Set_Mappy_Star (D);

	if (NC->nb_thread == 1) {NNFC_Modele(D);}
	if (NC->nb_thread > 1) {NNFC_Modele_Thread(D);}
	NNFC_DEL_Data (D);

    return 0;
}