#ifndef BYNETWORK_H
#define BYNETWORK_H


#include <iostream>
#include <vector>
#include <map>
#include <math.h>  
using namespace std;

class BysCl{
public:
	BysCl(int numLabels, int numFeatures){
		nl=numLabels;
		nf=numFeatures;
		

		for (int i=0;i<nl;i++){
			priorP_p.push_back(0);
			priorP_ct.push_back(0);

			vector<double> v;//row
			for (int i=0;i<nf;i++){
				v.push_back(0);
			}
			pixelP_ct.push_back(v);pixelP_pl.push_back(v);

		}



	}
	
	int nl; int nf;
	std::vector<double> priorP_p;
	std::vector<double> priorP_ct;//count
	vector<vector<double>> pixelP_ct;
	vector<vector<double>> pixelP_pl;


	vector<vector<double>> GetPixelP(){
		return pixelP_pl;
	}
	vector<double> GetPriorP(){
		return priorP_p;
	}

	void calcP(vector<unsigned char> lb, vector<vector<unsigned char>> ti){//trainlabels,trainimages 
		
		//calc prior p
		for (int i=0;i<lb.size();i++){
			priorP_ct[static_cast<int>(lb[i])]++;
			priorP_p[static_cast<int>(lb[i])]+=1/((double)lb.size());

		}	

		//calc conditional p for pixels	
		for (int i=0;i<ti.size();i++){//for every image
			for (int j=0;j<nf;j++){//for every features
				if (ti[i][j]==1){//when white
					pixelP_ct[static_cast<int>(lb[i])][j]++;				
				}
			}
		}



		for (int i=0;i<nl;i++){//grid nfxnl
			for (int j=0;j<nf;j++){
				pixelP_pl[i][j]=((double)pixelP_ct[i][j]+1)/((double)priorP_ct[i]+2);
				double q1=((double)pixelP_ct[i][j]+1)/((double)priorP_ct[i]+2);
				double q0=1-q1;
				
			}	
		}


		

	}
	int classify(std::vector<unsigned char> t){
		int maxc=0;
		double max;
		double sump=0;

			for (int j=0;j<nf;j++){
				if (static_cast<int>(t[j])==1){
					sump+= log (pixelP_pl[0][j]);
				}
				else {
					sump+= log (1 - pixelP_pl[0][j]);
				}
			}	

		max=sump + log (priorP_p[0]);

		for (int i=1;i<nl;i++){
			sump=0;
			for (int j=0;j<nf;j++){
				if (static_cast<int>(t[j])==1){
					sump+= log (pixelP_pl[i][j]);
				}
				else {
					sump+= log (1 - pixelP_pl[i][j]);
				}
			}
			sump+= log (priorP_p[i]);
			if (max<=sump){
				max=sump;
				maxc=i;
			}
		
		}
		return maxc;

		
	}



	vector<vector<int>> elv(vector<vector<unsigned char>> ti, vector<unsigned char> tl){
		std::vector<vector<int> > mt;
		for (int i=0;i<nl;i++){
			vector<int> v;//row
			for (int i=0;i<nl;i++){
				v.push_back(0);
			}
			mt.push_back(v);
		}

		for (int i=0;i<ti.size();i++){
			int k=classify(ti[i]);
			mt[static_cast<int>(tl[i])][k]++;
		}


		return mt;

	}



};









#endif