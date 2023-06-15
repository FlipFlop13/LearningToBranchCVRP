#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <string.h>
#include <cstring>
#include <sstream>
#include <random>
#include <filesystem>
using namespace std;
namespace fs = filesystem;

/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of integer to be displayed in the console.
void printIntVector(vector<int> vec)
{
  cout << endl
       << "Vector:" << endl;
  for (int i : vec)
  {
    cout << i << endl;
  }
}
/// @brief This function prints every string element in the vector to the console.
/// @param vec A vector of string to be displayed in the console.
void printStringVector(vector<string> vec)
{

  cout << endl
       << "Vector:" << endl;
  for (string i : vec)
  {
    cout << i << endl;
  }
}
/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of vectors of integers to be displayed in the console.
void printIntVector2D(vector<vector<int>> vec)
{
  cout << endl
       << "Vector:" << endl;
  for (vector<int> i : vec)
  {
    for (int j : i)
    {
      cout << j << " ";
    }
    cout << endl;
  }
}
/// @brief This function prints every string element in the vector to the console.
/// @param vec A vector of vectors of strings to be displayed in the console.
void printStringVector2D(vector<vector<string>> vec)
{
  cout << endl
       << "Vector:" << endl;
  for (vector<string> i : vec)
  {
    for (string j : i)
    {
      cout << j << " ";
    }
    cout << endl;
  }
}
/// @brief This function generates a vecotr containing all the files in a directory with the given filetype.
/// @param path Path to the directory, default : ./Graphs/X/
/// @param filetype The filetype, the function will return only this type. e.g. pdf. default:* (all filetypes)
/// @return A vector of strings containing all the filenames.
vector<string> glob(string path = "./Graphs/X/", string filetype = "*")
{
  vector<string> filenames;
  int lenghtFileType = filetype.length();
  for (const auto &entry : fs::directory_iterator(path))
  {
    string filename = entry.path();
    if (filetype == "*")
    {
      filenames.push_back(filename);
    }
    else
    {
      string thisFileType = filename.substr((filename.length() - (lenghtFileType)), lenghtFileType);
      if (thisFileType == filetype)
      {
        filenames.push_back(filename);
      }
    }
  }

  return filenames;
}

/// @brief Reads a vrp file and returns the values needed to generate the graph.
/// @param filename Path to the vrp file.
/// @return A tuple, containing a vector of a vector of two integers(x and y coordinate), a vector of demands, a vector for the depot location and an integer for the vehicle capacity.
tuple<vector<vector<int>>, vector<int>, vector<int>, int>
read(string filename)
{
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
          if (j == 0)
          {
            j++;
            continue;
          }
          int strToInt;
          strToInt = stoi(word);
          vecNCS[i][(j - 1)] = strToInt;
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
    // close the file object.
    newfile.close();
  }
  tuple<vector<vector<int>>, vector<int>, vector<int>, int> retTuple(vecNCS, vecDS, vecDepot, capacity);
  return retTuple;
}

vector<int> depotPositionGenerator(string depotPosition)
{
  vector<int> dp{0, 0};
  if (depotPosition == "C")
  {
    dp[0] = 500;
    dp[1] = 500;
  }
  else if (depotPosition == "R")
  {
    dp[0] = rand() % 1000;
    dp[1] = rand() % 1000;
  }
  else
  { // positioning is eccentric
    dp[0] = 0;
    dp[1] = 0;
  }
  return dp;
}

vector<vector<int>> generateCustomerCoordinates(int n = 100, string customerPositiong = "R")
{
  vector<vector<int>> customerCoordinates(n, vector<int>(2));
  for (int i = 0; i < n; i++)
  {
    customerCoordinates[i][0] = rand() % 1000;
    customerCoordinates[i][1] = rand() % 1000;
  }

  return customerCoordinates;
}

void generate(int n = 100, string depotPosition = "R", string customerPositioning = "R", string demandDistribution = "U")
{
  vector<vector<int>> vecNCS(1, vector<int>(2));
  vecNCS[0] = depotPositionGenerator("R");
  vector<vector<int>> tempV = generateCustomerCoordinates(n, customerPositioning);
  vecNCS.insert(vecNCS.end(), tempV.begin(), tempV.end());
  printIntVector2D(vecNCS)
}

int main()
{
  cout << "Starting" << endl;
  // tuple<vector<vector<int>>, vector<int>, vector<int>, int> cvrp;
  // cvrp = read("Graphs/X/X-n101-k25.vrp");
  // printIntVector2D(get<0>(cvrp));

  // string ft = "vrp";
  // string directory = "./Graphs/X/";
  // vector<string> g = glob(directory, ft);
  // printStringVector(g);

  generate();
  return 0;
}