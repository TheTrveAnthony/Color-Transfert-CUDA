#include <iostream>
#include "function.h"

 
 
using namespace std;


 
 
int main(int argc, char **argv)
{

  if (argc != 3) {

  	cout << "Enter the name of the pic you wanna make up and the one it will take its new colors from" << endl ;
  	return 0;

  }

  string n1(argv[1]);
  string n2(argv[2]);

  transfert(n1, n2);

    return 0;
}