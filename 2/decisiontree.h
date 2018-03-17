#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>

using namespace std;

class Treenode{
public:	
	Treenode(){

		parent=NULL;
		plu=false;
	}

	bool plu;
	string outputval;
	Treenode* parent;
	std::vector<Treenode*> kidnodes;
	int attr;//feature indices
};

#endif