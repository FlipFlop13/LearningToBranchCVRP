#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring> 
#include <sstream>
using namespace std;

int main(){
   vector<vector<int>> vecNCS( 101 , vector<int> (3));
   vector<int> vecDS(101 ,0);
   fstream newfile;
   newfile.open("Graphs/X/X-n101-k25.vrp",ios::in); //open a file to perform read operation using file object
   if (newfile.is_open()){ //checking whether the file is open
      string tp;
      string ncs  = "NODE_COORD_SECTION";
      bool bncs = false;
      string ds = "DEMAND_SECTION";
      bool bds = false;
      int i = 0;
      cout << "i:" << i << endl;
      while(getline(newfile, tp)){ //read data from file object and put it into string.
          cout << i << endl;
          int j = 0;
          stringstream ss(tp);  
          string word; 
          while (ss >> word) { // Extract word from the stream.

              if (word == ncs){
                bncs = true;
                i = -1;
                continue;
              }
              if (word == ds){
                bds = true;
                bncs = false;
                i = -1;
                continue;
              }
              if (bncs){
                  int strToInt;
                  cout << word << "  ";
                  strToInt = stoi(word);
                  vecNCS[i][j] = strToInt;
                  
              }
              
              if (bds){
                  int strToInt;
                  cout << word << "  ";
                  strToInt = stoi(word);
                  vecDS[i] = strToInt;
              }
              









              j++;
          }
        i++;
        cout << endl;
      }

    
    cout << endl;
      newfile.close(); //close the file object.
   }
}