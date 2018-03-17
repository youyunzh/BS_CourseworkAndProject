#ifndef SAMPLE_H
#define SAMPLE_H

#include <string>
#include <cassert>
#include <iostream>
#include <vector>
#include <stdlib.h> 
#include <sstream>
using namespace std;

class Sample {
public:
	Sample(std::vector<string> v, int id){
		string g=v[0];
		stringstream ss(g);
		cl=0;
		ss>>cl;

    	v.erase(v.begin());
	
		for (int i=0; i<v.size();i++){
			features.push_back(v[i]);
		}
		index=id;
	}

	vector<string> features;
	bool cl;
	int index; //sample index
};
#endif