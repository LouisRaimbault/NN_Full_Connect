#ifndef GENERAL_H
#define GENERAL_H

#include <fstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <random>
#include <ctype.h>


struct Grad
{
	float ** pt_Coeff;
	float ** pt_dCoeff;
	float * dCoeff_updt;
	float * G1;
	float * G2;
	float p1;
	float p2;
	int nb_coeff;
	float lr; 
	float Epsilon;
	float cur_epoch;
};


struct Mappy_Star
{
	std::unordered_map<std::string,float(*)(float )> * map_Activ;
	std::unordered_map<std::string,float(*)(float )> * map_deri_Activ;

	std::unordered_map<std::string,float(*)(float *, int)> * map_Cost;
	std::unordered_map<std::string,void(*)(float *, float *, int)> * map_deri_Cost;

	std::unordered_map<std::string,void(*)(float *, float *, int)> * map_Quality_Coeff;
	std::unordered_map<std::string,void(*)(Grad *)> * map_updt_LR;

	std::unordered_map<std::string,int>  * map_Alpha_Beta;

};

struct NNFC_config
{
  char * Path_Config; 
	std::string  * NomTarget;
	int nb_Layer;
	int * nb_Nodes_Layer;
	float * Range_First_Teta;
	std::string * Name_F_Activ;
	std::string  * Name_F_Cost;
	float pcent_Test;
	float Learate;
	float * Values_Hyparam;
	std::string  * Name_F_Learate;
	int nb_epoch;
	std::string * Name_F_Quality;
	int do_normalization;
	int nb_thread;
	Mappy_Star * MS;
};



struct Layer
{
	float * Input;
	float * Poids;
	float * dInput;
	float * dPoids;
	float * Z;
	int nb_in;
	int nb_out;
	float (*F_Activ)(float);
	float (*dF_Activ)(float);
	float * Alpha_Beta; // en fonction de si la derivee utilise f(z) (beta) ou z (alpha) pour se calculer 
	Layer * Father;
	Layer * Son;

};


struct NN_Full_Connect
{
	float (*F_Cost)(float *, int);
	void (*dF_Cost)(float *, float *, int);
	void (*F_updt_lr)(Grad *);
	void (*F_Quality)(float *, float *, int);
	Layer ** Tab_Layer;
	int nb_Layer;
	Grad * G;
	int nb_epoch;
	int nb_val_Fwd;
	float * Val_Fwd;
	float cur_Y;
	float learate;
};

struct Base 
{
	int nb_ind;
	int nb_var;
	float ** X;
	float * Y; // Faire +3 pour les indicateurs 
	float * Y_hat;
	// Ajouter les mesure de qualités 
};

struct Info_Thread
{
	NN_Full_Connect ** Th_NNFC;
	Grad ** Th_Grad;
	int ** Th_ind;
	int * Th_nb_ind;
};

struct Data
{
	NNFC_config * NC;
	NN_Full_Connect * NNFC;
	Base * Train;
	Base * Test;
	std::string * Nomvar;
	std::string * NomTarget;
	std::string * nomModY;
	float ** X; // Dim1 = lignes, Dim2 = Colonnes
	float * Y;
	float avg_Y;
	float sd_Y;
	float * avg_X;
	float * sd_X;
	int nb_mod_Y;
	int nb_ind; 
	int nb_var;
	int nb_thread;
	Info_Thread * ITh;
}; 



void NNFC_DEL_Mappy_Star (Mappy_Star * MS);
void NNFC_DEL_Config (NNFC_config * NC);
void NNFC_DEL_Base (Base * B);
void NNFC_DEL_Data (Data * D);
void NNFC_DEL_Grad (Grad * G);
void NNFC_DEL_Layer (Layer * L);
void NNFC_DEL_NN_Full_Connect (NN_Full_Connect * NNFC);
void NNFC_Print_Config(NNFC_config * NC);
void NNFC_Print_Data (Data * D, int n);
void fill_arr_from_file(char * pathfile, uint64_t * l, int * nrow, char ** tabchar);
void NNFC_Get_Config (NNFC_config * NC);
std::string findKeyByValue( std::unordered_map<std::string, int> * map, int value);
void NNFC_Get_Data (Data * D, char * path_data, char sep);

#endif