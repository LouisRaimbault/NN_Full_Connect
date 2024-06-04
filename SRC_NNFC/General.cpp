#include "General.h"

void NNFC_DEL_Mappy_Star (Mappy_Star * MS)
{
  delete MS->map_Activ;
  delete MS->map_deri_Activ;
  delete MS->map_Cost;
  delete MS->map_deri_Cost;
  delete MS->map_Quality_Coeff;
  delete MS->map_updt_LR;
  delete MS->map_Alpha_Beta;
  free(MS);
}

void NNFC_DEL_Config (NNFC_config * NC)
{
  free(NC->nb_Nodes_Layer);
  free(NC->Range_First_Teta);
  free(NC->Values_Hyparam);
  delete [] NC->Name_F_Activ;

  delete NC->Name_F_Cost;
  delete NC->NomTarget;
  delete NC->Name_F_Learate;
  delete NC->Name_F_Quality;
  NNFC_DEL_Mappy_Star (NC->MS);
  free(NC);
}

void NNFC_DEL_Base (Base * B)
{
  free(B->X);
  free(B->Y);
  free(B->Y_hat);
  free(B);
}


void NNFC_DEL_Grad (Grad * G, int b) // b for numthread
{
  if (b == 0)
  {
    free(G->G1);
    free(G->G2);
    free(G->pt_Coeff);
    
  }
  free(G->pt_dCoeff);
  free(G->dCoeff_updt); 
  free(G);

}

void NNFC_DEL_Layer (Layer * L, int a, int b, Data * D) // a num Layer, b num thread
{
  if (b == 0)
  {
    if (a != 0)
    { 
      free(L->Input);
      free(L->dInput);
    }
    if (a != D->NC->nb_Layer-1)
      {
        free(L->Z);
        free(L->Poids);
        free(L->dPoids);
      }
    
    free(L);
    return;
  }

  if (a != 0)
    {
      free(L->Input);
      free(L->dInput);
    }
  if (a != D->NC->nb_Layer-1)
    {
      free(L->dPoids);
      free(L->Z);
    }  
  
  free(L);

}

void NNFC_DEL_NN_Full_Connect (NN_Full_Connect * NNFC, int b ,Data * D) // b for numthread
{
  for (int i = 0; i < NNFC->nb_Layer;i++)
    {
      NNFC_DEL_Layer(NNFC->Tab_Layer[i],i,b,D);

    }
  NNFC_DEL_Grad(NNFC->G,b);
  free(NNFC->Tab_Layer);
  free(NNFC);
}

void NNFC_DEL_Info_Thread (Info_Thread * ITh, Data * D)
{
  free(ITh->Th_ind[0]);
  for (int i = 1; i < D->nb_thread; i++)
    {
      NNFC_DEL_NN_Full_Connect (ITh->Th_NNFC[i], i ,D);
    }

  free(ITh->Th_nb_ind);
  free(ITh->Th_NNFC);
  free(ITh->Th_Grad);
  free(ITh);

}

void NNFC_DEL_Data (Data * D)
{
  NNFC_DEL_Base (D->Train);
  NNFC_DEL_Base (D->Test);
  NNFC_DEL_NN_Full_Connect(D->NNFC,0,D);
  if (D->NC->nb_thread > 1)
    {
      NNFC_DEL_Info_Thread (D->ITh,D);
    }
  if (D->NC->do_normalization == 1)
  {  
    free(D->avg_X);
    free(D->sd_X);
  }
  NNFC_DEL_Config(D->NC);
  delete [] D->Nomvar;
  if (D->nb_mod_Y > 1) {delete [] D->nomModY;}
  free(D->Y);
  for (int i = 0; i < D->nb_ind; i++) {free(D->X[i]);} free(D->X);

  free(D);
}


void NNFC_Print_Config(NNFC_config * NC)
{
  std::cout << "################################################### \n\n";
  std::cout << "Configuration du Reseaux de Neurones Full Connected \n";
  std::cout << "Target : " << *(NC->NomTarget) << "\n";
  std::cout << "Nb Layers : " << NC->nb_Layer << "\n";
  std::cout << "Nb Ner/Layer : ";
  for (int i = 0; i < NC->nb_Layer; i++) {std::cout << "[" << NC->nb_Nodes_Layer[i] << "] ";} std::cout << "\n";
  std::cout << " Range First Teta : ";
  for (int i = 0; i < 2; i++) {std::cout << NC->Range_First_Teta[i] << "\t";} std::cout << "\n";
  std::cout << "Activation Functions/Layers : ";
  for (int i = 0; i < NC->nb_Layer; i++) {std::cout << "[" << NC->Name_F_Activ[i] << "] ";} std::cout << "\n";
  std::cout << "Const Function [" << *(NC->Name_F_Cost) << "]\n";
  std::cout << "Learnign Rate : " << NC->Learate << " With Update : " << *(NC->Name_F_Learate) << "\n";
  std::cout << "Hyper param for " <<  *(NC->Name_F_Learate) << " : ";
  for (int i = 0; i < 2; i++) {std::cout << NC->Values_Hyparam[i] << "\t";} std::cout << "\n";
  std::cout << "NB Epoch : " << NC->nb_epoch << "\n";
  std::cout << "Quality Function : " << *(NC->Name_F_Quality) << "\n"; 
  std::cout << "Nb ind Test = Nb_ind *  : " << NC->pcent_Test << "\n"; 
  std::cout << "Do Normalization : " << NC->do_normalization << "\n"; 
  std::cout << "NB Thread : " << NC->nb_thread << "\n";
  std::cout << "\n ################################################### \n\n";

}

void NNFC_Print_Data (Data * D, int n)
{
  std::cout << *(D->NomTarget);
  for (int j = 0; j < D->nb_var; j++)
    {
      std::cout << "\t" <<D->Nomvar[j]; 
    }
  std::cout << "\n";
  for (int i = 0; i < n; i++) 
    {
      std::cout << D->Y[i];
      for (int j = 0; j < D->nb_var; j++)
        {
          std::cout << "\t" << D->X[i][j];
        }
      std::cout << "\n";
    }

}

void fill_arr_from_file(char * pathfile, uint64_t * l, int * nrow, char ** tabchar)
{
  FILE *Fichier ;
  Fichier = fopen(pathfile,"r");
  uint64_t tr_l = 0 ;
  uint64_t i; 
  char c;
  int tr_nrow = 0;  
  if (Fichier != NULL)
    {                         // count number of lines and char
      while (!feof(Fichier))  
       {c = getc(Fichier);
        tr_l ++;
        if (c == '\n') tr_nrow++; }                      
    }
   fclose(Fichier); //close file
   tr_nrow++;
  char * tr_tabchar = (char*)malloc(tr_l * sizeof(char)); // create Buffer of char
  Fichier = fopen(pathfile,"r"); // open again file
  if ( Fichier != NULL && tr_tabchar  != NULL)
    {    
      // Fill array
     while (i < tr_l)  
      { c = getc(Fichier);
        tr_tabchar[i++]=c;
      }
     }
  else perror ("\n\n problem in file ");
  fclose(Fichier); // close file
  tr_tabchar[tr_l-1] = '\n';
  *(l) = tr_l;
  *(nrow) = tr_nrow;
  *(tabchar) = tr_tabchar;

}





void NNFC_Get_Config (NNFC_config * NC)
{
  uint64_t l;
  int i,nrow;
  char * Buffer;
  fill_arr_from_file(NC->Path_Config,&l,&nrow,&Buffer);
  i = 0; std::string st = "";
  while (Buffer[i] != ':') {i++;} i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Recupération du nom variable
  NC->NomTarget = new std::string (st); st = "";
  while (Buffer[i] != ':') {i++;} i++;
  while (Buffer[i] != ',') {st = st =Buffer[i]; i++;} i++; // Récupération du nombre de Layer
  NC->nb_Layer = std::stoi(st); st = "";
  NC->nb_Nodes_Layer = (int*)malloc(NC->nb_Layer*sizeof(int));
  NC->Name_F_Activ = new std::string [NC->nb_Layer]; 
  while (Buffer[i] != ':') {i++;} i++;
  for (int j = 0; j < NC->nb_Layer;j++) // Récupération du nombre de Neurones par Layer 
    {
      while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
      NC->nb_Nodes_Layer[j] = std::stoi(st); st = "";
    }

  NC->Range_First_Teta = (float*)malloc(2*sizeof(float)); // Récupération de la range pour les premiers coeffs
  while (Buffer[i] != ':') {i++;} i++;
  for (int j = 0; j < 2; j++)
    {
      while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
      NC->Range_First_Teta[j] = std::stof(st); st = "";     
    }

  while (Buffer[i] != ':') {i++;} i++;
  for (int j = 0; j < NC->nb_Layer; j++) // Récupération des Fonctions d'activations par couches
    {
      while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
      NC->Name_F_Activ[j] = st; st = "";
    }
  while (Buffer[i] != ':')  {i++;} i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++; // Récupération de la fonction de cout 
  NC->Name_F_Cost = new std::string(st); st = "";
  while (Buffer[i] != ':') {i++;} i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++; // Récupération du Learning Rate
  NC->Learate = std::stof(st); st = "";
  while (Buffer[i] != ':') {i++;} i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} // Récupération de la fonction d'optim Learning Rate
  NC->Name_F_Learate = new std::string(st); st = "";

  NC->Values_Hyparam = (float*)malloc(2*sizeof(float)); // Récupération des coeff pour l'optim Learning Rate
  while (Buffer[i] != ':') {i++;} i++;
  for (int j = 0; j < 2; j++)
    {
      while (Buffer[i] != ',') {st = st + Buffer[i]; i++;} i++;
      NC->Values_Hyparam[j] = std::stof(st); st = "";     
    }

  while (Buffer[i] != ':') {i++;} i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++;  // Récupération du nombre d'epoch
  NC->nb_epoch = std::stoi(st); st = "";
  while (Buffer[i] != ':') {i++;}i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++; // Récupération du print quality Modele
  NC->Name_F_Quality = new std::string (st); st = "";
  while (Buffer[i] != ':') {i++;}i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++; // Récupération de l'info :%ind pour Test 
  NC->pcent_Test = std::stof(st); st = "";
  while (Buffer[i] != ':') {i++;}i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++; // Récupération de l'info : normalisation ou non 
  NC->do_normalization = std::stoi(st); st = "";
  while (Buffer[i] != ':') {i++;}i++;
  while (Buffer[i] != ',') {st = st + Buffer[i]; i++;}i++; // Récupération de l'info : nombre de thread 
  NC->nb_thread = std::stoi(st); st = "";

  
  free (Buffer);
  NNFC_Print_Config(NC);
  

}

std::string findKeyByValue( std::unordered_map<std::string, int> * map, int value)
 {
    for (const auto& pair : *(map)) {
        if (pair.second == value) {
            return pair.first;
        }
    }
    return "";
}




void NNFC_Get_Data (Data * D, char * path_data, char sep)
{
  uint64_t l;
  int i,j,k,q,nrow,ncol,nb_ind,nb_var,nb_qualvar,nb_kantvar,repere; i = 0; ncol = 0;
  char * Buffer;
  fill_arr_from_file(path_data,&l,&nrow,&Buffer); 
  nb_ind = nrow-1;
  while (Buffer[i] != '\n') {if (Buffer[i] == ',') {ncol = ncol+1;} i++;} ncol = ncol+1;
  i = 0;
  std::string * Nomvar = new std::string [ncol];
  int * is_qual = (int*)calloc(ncol,sizeof(int));
  int * is_target = (int*)calloc(ncol,sizeof(int));
  std::string st = ""; j = 0;
  while (Buffer[i] != '\n') // Récupération des noms de variables
    {
      if (Buffer[i] == sep) {Nomvar[j] = st; st = ""; i++; j++; continue;}
      st = st + Buffer[i]; i++;
    } 
  Nomvar[ncol-1] = st;
  i++; repere = i; j = 0; q = 0; k = 0; char c = Buffer[i];
  while (Buffer[i] != '\n') // Repérer les var quanti et quali 
    {
      if (Buffer[i] == sep) {is_qual[j] = 1-isdigit(c); j++; i++; c = Buffer[i];}
      i++;  
    }
  is_qual[ncol-1] = 1-isdigit(c);
  for (j = 0; j < ncol; j++)// settage des Var Init
    {
      if (Nomvar[j] == *(D->NomTarget)) {is_target[j] = 1;}
      else {is_target[j] = 0;}
      if (is_qual[j] == 0)
        {
          k = k+1;
          continue;
        }
      q = q+1;
    }
  nb_qualvar = q; nb_kantvar = k;  
  float ** Mat_Kanti;
  if (k > 0 )
    { Mat_Kanti = (float**)malloc(k *sizeof(float*));
      for (j = 0; j < k; j++) {Mat_Kanti[j] = (float*)malloc(nb_ind*sizeof(float));}
    }  
  std::unordered_map<std::string,int> * Tab_Mappy;
  int * nb_mod_qual ;
  int ** num_mod ;

  if (q > 0)
    {
      nb_mod_qual = (int*)calloc(q,sizeof(int));
      num_mod = (int**)malloc(q*sizeof(int*));
      Tab_Mappy = new std::unordered_map<std::string,int> [q];
      for (j = 0; j < q; j++) {num_mod[j] = (int*)malloc(nb_ind*sizeof(int));nb_mod_qual[j] = 1;}
    }   
  
  i = repere; j = 0; st = "";
  for (int n = 0; n < nb_ind; n++) // Creation des vecteurs de données pour quali et quanti séparément
    {
      j = 0; k = 0; q = 0; 
      while (Buffer[i] != '\n')
        {
          if (Buffer[i] == sep)
            {
              if(is_qual[j] == 0)
                {
                  //std::cout << st << "\t";
                  Mat_Kanti[k][n] = std::stof(st);
                  k = k+1;
                  st = "";
                  i++; j++;
                  continue;
                }
              //std::cout << st << "\t";
              if (Tab_Mappy[q][st] == 0) {Tab_Mappy[q][st] = nb_mod_qual[q]; nb_mod_qual[q] = nb_mod_qual[q]+1;}
              num_mod[q][n] = Tab_Mappy[q][st]-1;
              st = "";
              q=q+1;
              i++; j++;
              continue;
            }
          st = st + Buffer[i];  
          i++;
        }
      if (is_qual[ncol-1] == 0)
        {
          //std::cout << st << "\n";
          Mat_Kanti[k][n] = std::stof(st);
          st = "";
          i++; 
          continue;        
        }
      //std::cout << st << "\n";
      if (Tab_Mappy[q][st] == 0) {Tab_Mappy[q][st] = nb_mod_qual[q]; nb_mod_qual[q] = nb_mod_qual[q]+1;}
        num_mod[q][n] = Tab_Mappy[q][st]-1;
        i++; j++; st = "";
        continue;
    }
  free(Buffer);
  float * Y = (float*)malloc(nb_ind*sizeof(float));
  int nb_mod_Y = 1; q = 0; k = 0;
  for (j = 0; j < ncol; j++)
    {
      if (is_target[j] == 0)  
        {
          if (is_qual[j] == 1) {q++; continue;}
          k++;
          continue;
        }
      if (is_qual[j] == 0)
        {
          for (i = 0; i < nb_ind; i++)
            {
              Y[i] = Mat_Kanti[k][i];
            }  
          break;
        }
      for (i = 0; i < nb_ind; i++)
        {
          Y[i] = (float)(num_mod[q][i]);          
        }  
      nb_mod_Y = nb_mod_qual[q]-1; 
      std::string * nomModY = new std::string [nb_mod_Y];
      for (int z = 0; z < nb_mod_Y;z++)
      {nomModY[nb_var] = findKeyByValue(&Tab_Mappy[q],z+1);} 
      D->nomModY = nomModY;  
      break;
    }  

  nb_var = nb_kantvar - nb_qualvar - nb_mod_Y; // Recupération du nombre total de variables, - qualvar car nb start à 1
  for (q = 0; q < nb_qualvar; q++) {nb_var = nb_var + nb_mod_qual[q];}
  // DIM 1 = Rows, DIM2 = VAR
  std::string * Nomvar_out = new std::string [nb_var];
  float ** MAT = (float**)malloc(nb_ind*sizeof(float*));
  for (j = 0; j < nb_ind; j++)  
    {
      MAT[j] = (float*)malloc((nb_var+1)*sizeof(float));
      for (i = 0; i < nb_var;i++) {MAT[j][i] = 0.0f;}
      MAT[j][i] = 1.0f;  
    } 
  q = 0; k =0; nb_var = 0;
  for (j = 0; j < ncol; j++) // Préparation de la matrice de données 
    {
      if (is_target[j] == 1)
        {
          if (is_qual[j] == 1) {q=q+1; continue;}
          k=k+1; continue;
        }
      if (is_qual[j] == 0)
        {
          float * pt = Mat_Kanti[k];
          Nomvar_out[nb_var] = Nomvar[j];
          for (i = 0; i < nb_ind; i++) {MAT[i][nb_var] = pt[i];}
          k=k+1;
          nb_var = nb_var+1;
          continue;
        }
      int * pt2 = num_mod[q];
      int nbm = nb_mod_qual[q]-1;
      repere = nb_var; 
      for (int z = 0; z < nbm; z++)
        {
          Nomvar_out[nb_var] = Nomvar[j] + ".";
          Nomvar_out[nb_var] = Nomvar_out[nb_var] + findKeyByValue(&Tab_Mappy[q],z+1);
          nb_var = nb_var+1;
        }
      for (i = 0; i < nb_ind; i++)
        {
          MAT[i][repere+pt2[i]] = (float)1.00;
        }
      q = q+1;  
    }


  
  delete [] Nomvar;
  free(is_target);
  free(is_qual);  
  if (nb_qualvar > 0) {delete [] Tab_Mappy; free(nb_mod_qual); for (q = 0; q < nb_qualvar; q++) {free(num_mod[q]);} free(num_mod);}
  if (nb_kantvar > 0) {for (k = 0; k < nb_kantvar; k++) {free(Mat_Kanti[k]);} free(Mat_Kanti);}
  D->Nomvar = Nomvar_out;
  D->X = MAT;
  D->Y = Y;
  D->nb_mod_Y = nb_mod_Y;
  D->nb_ind = nb_ind;
  D->nb_var = nb_var;
}







