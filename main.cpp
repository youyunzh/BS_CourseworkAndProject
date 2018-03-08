#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "sample.h"
#include "decisiontree.h"
#include "makedecision.h"
#include <algorithm>

using namespace std;

int main (int argc, char *argv[]) {
  std::vector<string> featureNames;
  std::vector<int> attributes;
  std::vector<Sample> sampleset;
  //readfile & parse	
  string line;
  string filename(argv[1]);
  ifstream myfile (filename);
  if (myfile.is_open()){
  	getline (myfile,line);
   
    stringstream ss(line);
    string attr;
    while(getline(ss, attr,',')){
    	featureNames.push_back(attr);

    }	
    int id=1;
    while (getline(myfile,line)){
    	std::vector<string> v;
    	stringstream ss(line);
    	string val;
    	while(getline(ss, val,',')){
    		v.push_back(val);
    	}


    	Sample curr(v, id);id++;
    	sampleset.push_back(curr);
    }
    myfile.close();
    for (int i=0;i<featureNames.size()-1;i++){
    	attributes.push_back(i);
    }

 

    random_shuffle(sampleset.begin(),sampleset.end());
	std::vector<Sample> training;
	std::vector<Sample> validation;
	std::vector<Sample> testing;

	for (int i=0;i<sampleset.size();i++){
		if (i<sampleset.size()*3/5){
			training.push_back(sampleset[i]);
		}
		else if(i<sampleset.size()*4/5){
			validation.push_back(sampleset[i]);
		}
		else{
			testing.push_back(sampleset[i]);
		}
	}



	Treenode* r;

	double max_tra_acu=-1.0;int max_tra_dep=-1;
	double max_val_acu=-1.0;int max_val_dep=-1;
	double max_tes_acu=-1.0;int max_tes_dep=-1;

	cout<<"depth train\%     valid\%"<<endl;
	for (int i=1;i<16;i++){//i is the depth

		r=makeTree(training,attributes,training, attributes.size()-i );
		int ct_tra=0;
		int ct_val=0;
		
		for (int i=0;i<training.size();i++){
			bool predi= predict(r, training[i]);
			if (predi==training[i].cl){
				ct_tra++;
			}
		}

		for (int i=0;i<validation.size();i++){
			bool predi= predict(r, validation[i]);
			if (predi==validation[i].cl){
				ct_val++;
			}
		}



		double tra_per=(1.0*ct_tra)/(1.0*training.size())*100;
		double val_per=(1.0*ct_val)/(1.0*validation.size())*100;
		
		cout<<i<<"     "<<tra_per<<"     "<<val_per<<endl;

		if (max_tra_acu<tra_per){
			max_tra_acu=tra_per;
			max_tra_dep=i;
		}
		if (max_val_acu<val_per){
			max_val_dep=i;
			max_val_acu=val_per;
		}



	}

	cout<<"max depth of training set is: "<<max_tra_dep<<" with an accuracy "<<max_tra_acu<<endl;
	cout<<"max depth of validation set is: "<<max_val_dep<<" with an accuracy "<<max_val_acu<<endl;
	std::vector<Sample> traPlusVal;
	traPlusVal=training;
	for (int i=0;i<validation.size();i++){
		traPlusVal.push_back(validation[i]);
	}
	
	
	r=makeTree(traPlusVal,attributes,traPlusVal,attributes.size()-max_val_dep);


	int ct_tes=0;
	for (int i=0;i<testing.size();i++){
		bool predi= predict(r, testing[i]);
		if (predi==testing[i].cl){
			ct_tes++;
		}
	}
	double tes_per=(1.0*ct_tes)/(1.0*testing.size())*100;
	cout<<"accuracy of testing set using depth "<<max_val_dep<<"is "<<tes_per<<endl;



  }

  else cout << "Unable to open file"; 

  return 0;
}

