CXX = g++

CXXFLAGS = -mpopcnt -msse4 -mabm -mmmx -mavx -mavx -mavx2 -mfma -O3 
CXXFLAGS2 = -lpthread -mpopcnt -msse4 -mabm -mmmx -mavx -mavx -mavx2 -mfma -O3 

NNFC: Apply_Modele.o Apply_Modele_init.o NNFC_modele_thread.o NNFC_modele.o NNFC_init_modele.o Dico_Function.o General.o Order.o NNFC_main.o
	$(CXX) Apply_Modele.o Apply_Modele_init.o NNFC_modele_thread.o NNFC_modele.o NNFC_init_modele.o Dico_Function.o General.o Order.o NNFC_main.o $(CXXFLAGS) -o NNFC 


Apply_Modele.o: Apply_Modele.cpp Apply_Modele.h Apply_Modele_init.h NNFC_modele.h General.h
	$(CXX) -c Apply_Modele.cpp $(CXXFLAGS2)
Apply_Modele_init.o: Apply_Modele_init.cpp Apply_Modele_init.h General.h
	$(CXX) -c Apply_Modele_init.cpp $(CXXFLAGS2)
NNFC_modele_thread.o: NNFC_modele_thread.cpp NNFC_modele_thread.h NNFC_modele.h NNFC_init_modele.h General.h
	$(CXX) -c NNFC_modele_thread.cpp $(CXXFLAGS2)
NNFC_modele.o: NNFC_modele.cpp NNFC_modele.h NNFC_init_modele.h General.h
	$(CXX) -c NNFC_modele.cpp $(CXXFLAGS)
NNFC_init_modele.o: NNFC_init_modele.cpp NNFC_init_modele.h General.h
	$(CXX) -c NNFC_init_modele.cpp $(CXXFLAGS)
Dico_Function.o: Dico_Function.cpp Dico_Function.h Order.h General.h
	$(CXX) -c Dico_Function.cpp $(CXXFLAGS)
General.o: General.cpp General.h
	$(CXX) -c General.cpp $(CXXFLAGS)
Order.o: Order.cpp Order.h General.h
	$(CXX) -c Order.cpp $(CXXFLAGS)
NNFC_main.o: NNFC_main.cpp Apply_Modele.h Apply_Modele_init.h NNFC_modele_thread.h NNFC_modele.h Dico_Function.h General.h 
	$(CXX) -c NNFC_main.cpp $(CXXFLAGS) 
