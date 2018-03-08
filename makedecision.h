#ifndef MAKEDECISION_H
#define MAKEDECISION_H
#include "sample.h"
#include "decisiontree.h"
#include <math.h>
using namespace std;

bool ifsameclass(vector<Sample> x){
	bool cls=x[0].cl;
	for (int i=1;i<x.size();i++){
		if (x[i].cl!=cls){
			return false;
		}
	}
	return true;
}
bool plurality(vector<Sample> x){
		int count=0;
		for (int i=0;i<x.size();i++){
			if (x[i].cl==1){count++;}
		}

		if(count>=x.size()/2){
			return true;	
		}
		else {
			return false;
		}
}

vector<string> featureval (int a, std::vector<Sample> x){//what are the values a feature has
	bool search=0;
	std::vector<string> featureval;
	for (int i=0;i<x.size();i++){
		search=0;
		for (int j=0;j<featureval.size();j++){
			if (x[i].features[a]==featureval[j]){
				search=1;
				break;
			}
		}
		if (search==0){
			featureval.push_back(x[i].features[a]);
		}
	}
	return featureval;
}

double importance(int a, std::vector<Sample> x){
	vector<vector<int> > branch;
	std::vector<string> featureval;

	bool search=0;
	for (int i=0;i<x.size();i++){
		search=0;
		

		for (int j=0;j<featureval.size();j++){

			if (x[i].features[a]==featureval[j]){	
				search=1;
				if (x[i].cl==0){
					branch[j][0]++;
				}
				else {
					branch[j][1]++;
				}
			}
		}
		if (search==0){
			std::vector<int> v;
			v.push_back(0);v.push_back(0);//one for class true, one for false
			branch.push_back(v);
			featureval.push_back(x[i].features[a]);
			if (x[i].cl==0){
				branch[branch.size()-1][0]++;
			}
			else {
				branch[branch.size()-1][1]++;
			}
		}
	}


	double imp=0.0;	
	for (int i=0;i<branch.size();i++){
		
		
		double prob = (1.0*branch[i][0]) / (1.0*branch[i][0]+1.0*branch[i][1]);
		double ent=0;
		if (prob==0 || prob==1){//can't handle log0, therefore cases are experated
			ent = 0;
		}
		else {
			ent = (-1.0)*(prob* (log2 (prob)) + (1.0-prob)* (log2 (1-prob)));
		}	
		double b = (1.0*branch[i][0]+1.0*branch[i][1])/(1.0*x.size());

		double add=b*ent;
		imp+= b * ent;

	}

	return (1-imp);

}

int maximpind(vector<int> f, vector<Sample> x){//return max-important-index
	double maximp=-1.0;
	int f_id=-1;//feature index
		for (int i=0;i<f.size();i++){
		
			double fi=importance(f[i],x);
			if (maximp<fi){
				maximp=fi;
				f_id=f[i];
			}	
		}
		
	return f_id;
}



Treenode* makeTree(vector<Sample> x, vector<int> f,vector<Sample> px, int dnum){
//dnum is the number of features - depth, meaning number of features the tree could expend on



	if (x.empty()){
		Treenode *curr=new Treenode;
		curr->plu=plurality(px);
		return curr;
	}
	else if (ifsameclass(x)){

		Treenode *curr=new Treenode;
		curr->plu=x[0].cl;
		return curr;
	}
	else if (f.empty()){
		Treenode *curr=new Treenode;
		curr->plu = plurality(x);
		return curr;
	}
	else {
		int attrid=maximpind(f,x); 
		
		Treenode* r = new Treenode();
		r->attr=attrid;
		std::vector<string> feaval;//feature values
		feaval=featureval(attrid,x);	
	

		bool plur=plurality(x);		
		r->plu=plur;


		for (int i=0; i< feaval.size(); i++){
			Treenode *curr=new Treenode;
			std::vector<Sample> kidx;
			for (int j=0; j<x.size();j++){
				if (x[j].features[attrid]==feaval[i]){
					kidx.push_back(x[j]);
				}
			}

			std::vector<int> kidf; 
			kidf=f;
			for (int k=0;k<kidf.size();k++){
				if (kidf[k]==attrid){
					kidf.erase(kidf.begin()+k);
				}
			}


			if (kidf.size()==dnum){
				kidf.clear();
			}


			curr=makeTree(kidx, kidf, x, dnum);
			
			curr->outputval=feaval[i];
			if (curr!=0){
				r->kidnodes.push_back(curr);
			}
		}

		Treenode* other=new Treenode;
		other->plu=plur;	
		r->kidnodes.push_back(other);
		

		return r;
	}
}

bool predict(Treenode* root, Sample x0){

	if (root->kidnodes.size()==0){ 		
		return root->plu;
	}


		bool search=0;
		for (int j=0; j<root->kidnodes.size();j++){
			

			if (x0.features[root->attr]==root->kidnodes[j]->outputval){	
				
				search=1;
				return predict(root->kidnodes[j],x0);

			}

		}
		if (search==0){
			
			return predict(root->kidnodes[root->kidnodes.size()-1],x0);
			
		}



}




#endif