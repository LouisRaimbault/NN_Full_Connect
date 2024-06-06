#include "Dico_Function.h"

// Section Function Activation

float Activ_None (float val)
{
  return val;
}

float deri_Activ_None (float val)
{
	return 1.0f;	
}

float Activ_Relu (float val)
{
	if (val < 0.00) {return 0.0f;}
	return val;
}

float deri_Activ_Relu (float val) // Est une ALpha 
{
	if (val < 0.00f) {return 0.0f;}
	return 1.0f;
}

float Activ_Leaky_Relu (float val)
{
	if (val < 0.00f) {return 0.01f * val;}
	return val;
}

float deri_Activ_Leaky_Relu (float val) // Est une alpha
{
	if (val < 0.00f) {return 0.01f;}
	return 1.0f;
}


float Activ_Sigmoide (float val)
{	
	return 1.0f/(1.0f+std::exp(-val));
}

float deri_Activ_Sigmoide (float val) // Est une Beta
{
	return val*(1-val);
}

float Activ_Tanh (float val) 
{
   return (std::exp(val)-std::exp(-val))/(std::exp(val)+std::exp(-val));
}

float deri_Activ_Tanh(float val) // Est une Beta
{
	return (1.0f-std::pow(val,2.0f));
}

// Section Funciton Cost 

float Cost_MSE (float * val, int nb_arg)
{
	 return std::pow((val[0]-val[1]),2);
}

void deri_Cost_MSE (float * val, float * deri_val ,int nb_arg)
{
  deri_val[0] =  2.0f*val[0] - 2.0f*val[1];	
}

float Cost_Cross_E (float * val, int nb_arg)
{
	return (*val);
	// A coder 
}

void deri_Cost_Cross_E (float * Val_Fwd, float * deri_val ,int nb_arg)
{

	float sum_exp = 0.0f;
	float * val = (float*)malloc(nb_arg*sizeof(float));
	for (int i = 0; i < nb_arg; i++) {val[i] = Val_Fwd[i];}
	int y = (int)(val[nb_arg-1]+0.1f);
	for (int i = 0; i < nb_arg-1; i++) {val[i] = 1.0f/(1.0f+std::exp(-val[i]));} 
	for (int i = 0; i < nb_arg-1; i++) {sum_exp = sum_exp + std::exp(val[i]);}
	float sft_max_Ok = std::exp(val[y])/sum_exp;	
	//float dcost = -(1.0f/sft_max_Ok);
	for (int i = 0; i < nb_arg-1;i++)
		{
			if (i == y) {deri_val[i] = (sft_max_Ok-1.0f)*(val[i]*(1.0f-val[i])); continue;}
			deri_val[i] = (std::exp(val[i])/sum_exp)*(val[i]*(1.0f-val[i])); 
		}

	free(val);
}

// Section Qualoity Coeff

void Qual_Pred (float * Y, float * Y_hat, int nb) // en val[1] le nombre de ligne de Y, ensuite et à la suite, Y et y hat 
{
  double SCE = 0.00;
  double SCT = 0.00;
  double SCR = 0.00;
  double MedErr = 0.00;
  double MAE = 0.00;
  double MSE = 0.00;
  double avg_vt = 0.0;


  float * ecart = (float*)malloc(nb*sizeof(float));
  for (int i = 0; i < nb; i++){avg_vt = avg_vt + Y[i];} avg_vt = avg_vt/(double)nb;
  std::cout << "avg_vt = " << avg_vt << "\n";
  for (int i = 0; i < nb; i++)
  	{
  		std::cout << "Y[i] = " << Y[i] << " et Y_hat[i] = " << Y_hat[i] << "\n";
      SCE = SCE + std::pow(std::abs(Y_hat[i]-avg_vt),2.00);
      SCT = SCT + std::pow(std::abs(Y[i]-avg_vt),2.00);
      SCR = SCR + std::pow(std::abs(Y[i]-Y_hat[i]),2.00);
      ecart[i] = std::abs(Y[i]-Y_hat[i]);
      MAE = MAE + std::abs(ecart[i]);
      MSE = MSE + (ecart[i])*(ecart[i]);
  	}

  special_float_order (ecart, nb);
  if (nb % 2 == 1) {MedErr = ecart[nb/2];}
  if (nb % 2 == 0) {MedErr = (ecart[nb/2]+ecart[nb/2])/2.0f;}  

  double R_2 = 1.0f - SCR/SCT;
  MAE = MAE/(double)nb;
  MSE = MSE/(double)nb;
  double RMSE = std::pow(MSE,0.5);

  std::cout << "SCE = " << SCE << "\n SCT = " << SCT << "\n SCR = " << SCR;
  std::cout << "\n MAE = " << MAE << "\n MSE = " << MSE << "\n RMSE = " << RMSE;
  std::cout << "\n MedErr = " << MedErr <<"\n R_2 = " << R_2 << "\n\n";

 

  free(ecart);
}


void Qual_Classif (float * Y, float * Y_hat, int nb)
{
	// En Qual Classif : Lignes vrai y  , Colones Prediction 
	int nb_mod = (int)(Y[nb]+0.001f);
	int ** Mat_Confuse = (int**)malloc(nb_mod*sizeof(int*));
	for (int i = 0; i < nb_mod; i++) {Mat_Confuse[i] = (int*)calloc(nb_mod,sizeof(int));}

	for (int i = 0; i < nb; i++) 
		{
			int l = (int)(Y[i]+0.01f);
			int c = (int)(Y_hat[i]+0.01f);
			Mat_Confuse[l][c] = Mat_Confuse[l][c]+1;
		}


	int nb_good = 0; int nb_false;
	for (int i = 0; i < nb_mod; i++) {nb_good = nb_good + Mat_Confuse[i][i];}
	nb_false = nb-nb_good;
	for (int i = 0; i < nb_mod; i++)
		{
			for (int j = 0; j < nb_mod; j++)
				{
					std::cout << "\t" << Mat_Confuse[i][j];
				}
			std::cout << "\n";
		}
	
	std::cout << "good : " << nb_good << " false : " << nb_false << " Ratio : " << (float)nb_good/(float)nb << "\n";	

	for (int i = 0; i < nb_mod;i++) {free(Mat_Confuse[i]);} free(Mat_Confuse);


}

// Section Update lr 

void updt_LR_None (Grad * G)
{
	for (int i = 0; i < G->nb_coeff; i++)
	{	
		*(G->pt_Coeff[i]) = *(G->pt_Coeff[i])- G->lr * G->dCoeff_updt[i];
	}
}


void updt_LR_Adagrad (Grad * G)
{
		for (int i = 0; i < G->nb_coeff; i++)
	{	
		G->G1[i] = G->G1[i] + std::pow(G->dCoeff_updt[i],2.00f); 
		*(G->pt_Coeff[i]) = *(G->pt_Coeff[i])- (G->lr/(std::pow(G->G1[i],0.5f)+G->Epsilon)) * G->dCoeff_updt[i];
	}
}


void updt_LR_Rmsprop (Grad * G)
{
		for (int i = 0; i < G->nb_coeff; i++)
	{	
		G->G1[i] = G->p1 * G->G1[i] + (1.0f - G->p1) * std::pow(G->dCoeff_updt[i],2.00f); 
		*(G->pt_Coeff[i]) = *(G->pt_Coeff[i])- (G->lr/(std::pow(G->G1[i],0.5f)+G->Epsilon)) * G->dCoeff_updt[i];
	}
}

void updt_LR_Adam (Grad * G)
{
	float cur_epoch = G->cur_epoch;
		for (int i = 0; i < G->nb_coeff; i++)
	{	
		G->G1[i] = G->p1 * G->G1[i] + (1.0f - G->p1) * G->dCoeff_updt[i]; 
		G->G1[i] = G->G1[i]/(1.0f-std::pow(G->p1,cur_epoch));
		G->G2[i] = G->p2 * G->G2[i] + (1.0f - G->p2) * std::pow(G->dCoeff_updt[i],2.0f); 
		G->G2[i] = G->G2[i]/(1.0f-std::pow(G->p2,cur_epoch));
		*(G->pt_Coeff[i]) = *(G->pt_Coeff[i])- (G->lr*G->G1[i])/(std::pow(G->G2[i],0.5f)+G->Epsilon);
	}
	G->cur_epoch = G->cur_epoch + 1.0f;
}


void Set_map_Activ (std::unordered_map<std::string,float(*)(float)> * map_Activ)
{
	map_Activ->operator[]("None") = Activ_None;
	map_Activ->operator[]("Relu") = Activ_Relu;
	map_Activ->operator[]("LeakyRelu") = Activ_Leaky_Relu;
	map_Activ->operator[]("Sigmoide") = Activ_Sigmoide;
	map_Activ->operator[]("Tanh") = Activ_Tanh;
}

void Set_map_deri_Activ (std::unordered_map<std::string,float(*)(float)> * map_deri_Activ)
{
  map_deri_Activ->operator[]("None") = deri_Activ_None;
	map_deri_Activ->operator[]("Relu") = deri_Activ_Relu;
	map_deri_Activ->operator[]("LeakyRelu") = deri_Activ_Leaky_Relu;
	map_deri_Activ->operator[]("Sigmoide") = deri_Activ_Sigmoide;
	map_deri_Activ->operator[]("Tanh") = deri_Activ_Sigmoide;
}

void Set_map_Cost (std::unordered_map<std::string,float(*)(float *, int)> * map_Cost)
{
	map_Cost->operator[]("MSE") = Cost_MSE;
	map_Cost->operator[]("CrossE") = Cost_Cross_E;	
}

void Set_map_deri_Cost (std::unordered_map<std::string,void(*)(float *, float * ,int)> * map_deri_Cost)
{
  map_deri_Cost->operator[]("MSE") = deri_Cost_MSE;
	map_deri_Cost->operator[]("CrossE") = deri_Cost_Cross_E;	
}

void Set_map_Quality_Coeff (std::unordered_map<std::string,void(*)(float *, float*, int)> * map_Quality_Coeff)
{
	map_Quality_Coeff->operator[]("Pred") = Qual_Pred;
	map_Quality_Coeff->operator[]("Classif") = Qual_Classif;	

}

void Set_map_updt_LR (std::unordered_map<std::string,void(*)(Grad *)> * map_updt_LR)
{
	map_updt_LR->operator[]("None") = updt_LR_None;
	map_updt_LR->operator[]("Adagrad") = updt_LR_Adagrad;
	map_updt_LR->operator[]("Rmsprop") = updt_LR_Rmsprop;
	map_updt_LR->operator[]("Adam") = updt_LR_Adam;

}

void Set_map_Alpha_Beta (std::unordered_map<std::string, int> * map_Alpha_Beta)
{
	map_Alpha_Beta->operator[]("None") = 0;
	map_Alpha_Beta->operator[]("Relu") = 0;
	map_Alpha_Beta->operator[]("LeakyRelu") = 0;
	map_Alpha_Beta->operator[]("Sigmoide") = 1;
	map_Alpha_Beta->operator[]("Tanh") = 1;
}

void Set_Mappy_Star (Data * D, int t)
{
	Mappy_Star * MS = (Mappy_Star*)malloc(sizeof(Mappy_Star));
	std::unordered_map<std::string,float(*)(float )> * map_Activ = new std::unordered_map<std::string,float(*)(float)>;
	std::unordered_map<std::string,float(*)(float )> * map_deri_Activ = new std::unordered_map<std::string,float(*)(float)>;

	std::unordered_map<std::string,float(*)(float *, int)> * map_Cost = new std::unordered_map<std::string,float(*)(float *,int)>;
	std::unordered_map<std::string,void(*)(float *, float *, int)> * map_deri_Cost = new std::unordered_map<std::string,void(*)(float *, float * ,int)>;

	std::unordered_map<std::string,void(*)(float *, float *, int)> * map_Quality_Coeff = new std::unordered_map<std::string,void(*)(float *, float *, int)>;
	std::unordered_map<std::string,void(*)(Grad *)> * map_updt_LR = new std::unordered_map<std::string,void(*)(Grad *)>;

	std::unordered_map<std::string,int> * map_Alpha_Beta = new std::unordered_map<std::string,int>;
	Set_map_Activ(map_Activ);
	Set_map_deri_Activ(map_deri_Activ);
	Set_map_Cost(map_Cost);
	Set_map_deri_Cost(map_deri_Cost);
	Set_map_Quality_Coeff(map_Quality_Coeff);
	Set_map_updt_LR (map_updt_LR);
	Set_map_Alpha_Beta(map_Alpha_Beta);


	MS->map_Activ = map_Activ;
	MS->map_deri_Activ = map_deri_Activ;
	MS->map_Cost = map_Cost;
	MS->map_deri_Cost = map_deri_Cost;
	MS->map_Quality_Coeff = map_Quality_Coeff;
	MS->map_updt_LR = map_updt_LR;
	MS->map_Alpha_Beta = map_Alpha_Beta;
	if (t == 0) {D->NC->MS = MS;}
	if (t == 1) {D->MC->MS = MS;}
}	
