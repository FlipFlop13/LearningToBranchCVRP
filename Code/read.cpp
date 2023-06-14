#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <cstring>
#include <sstream>
using namespace std;

tuple<vector<vector<int>>, vector<int>, vector<int>, int> read(string filename)
{
  // int read(string filename){

  vector<vector<int>> vecNCS(101, vector<int>(2));
  vector<int> vecDS(101, 0);
  vector<int> vecDepot{1, -1};
  int capacity = -1;

  fstream newfile;
  newfile.open(filename, ios::in); // open a file to perform read operation using file object
  if (newfile.is_open())
  { // checking whether the file is open
    string tp;

    bool bcap = false;
    bool bncs = false;
    bool bds = false;
    bool bDepotS = false;

    int i = 0;
    cout << "i:" << i << endl;
    while (getline(newfile, tp))
    { // read data from file object and put it into string.
      int j = 0;
      stringstream ss(tp);
      string word;
      while (ss >> word)
      { // Extract word from the stream.
        // Check if we are in the section for node location, node demand or depot location
        if (word == "CAPACITY")
        {
          bcap = true;
          bncs = false;
          bds = false;
          bDepotS = false;
        }
        if (word == "NODE_COORD_SECTION")
        {
          bcap = false;
          bncs = true;
          bds = false;
          bDepotS = false;
          i = -1;
          continue;
        }
        if (word == "DEMAND_SECTION")
        {
          bcap = false;
          bncs = false;
          bds = true;
          bDepotS = false;
          i = -1;
          continue;
        }
        if (word == "DEPOT_SECTION")
        {
          bcap = false;
          bncs = false;
          bds = false;
          bDepotS = true;
          i = -1;
          continue;
        }
        if (word == "EOF")
        {
          tuple<vector<vector<int>>, vector<int>, vector<int>, int> retTuple(vecNCS, vecDS, vecDepot, capacity);
          return retTuple;
        }

        if (bcap)
        {
          try
          {
            capacity = stoi(word);
          }
          catch (invalid_argument)
          {
            capacity = -1;
          }
          cout << "Capacity: " << capacity << endl;
        }
        if (bncs)
        {
          if (j==0){
            j++;
            continue;
          }
          int strToInt;
          strToInt = stoi(word);
          vecNCS[i][(j-1)] = strToInt;
        }
        if (bds)
        {
          int strToInt;
          strToInt = stoi(word);
          vecDS[i] = strToInt;
        }
        if (bDepotS)
        {
          int strToInt;
          cout << word << "  ";
          strToInt = stoi(word);
          vecDepot[i] = strToInt;
        }

        j++;
      }
      i++;
    }

    cout << endl;
    newfile.close(); // close the file object.
  }
  tuple<vector<vector<int>>, vector<int>, vector<int>, int> retTuple;
  return retTuple;
}

int main()
{
  tuple<vector<vector<int>>, vector<int>, vector<int>, int> cvrp;
  cvrp = read("Graphs/X/X-n101-k25.vrp");

  return 0;
}