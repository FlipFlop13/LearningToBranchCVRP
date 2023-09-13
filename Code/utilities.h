#ifndef UTILITIES_G // include guard
#define UTILITIES_G

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <string.h>
#include <math.h>
#include <cmath>
#include <cstring>
#include <sstream>
#include <random>
#include <filesystem>
#include <iomanip>
#include <array>
#include <map>
#include <algorithm>

using namespace std;

/// This file contains all the helper functions, including custom printer functions and the graph creator functions

/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of integer to be displayed in the console.
void printVector(vector<int> vec);
/// @brief This function prints every string element in the vector to the console.
/// @param vec A vector of string to be displayed in the console.
void printVector(vector<string> vec);
/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of vectors of integers to be displayed in the console.
void printVector(vector<vector<int>> vec);
/// @brief This function prints every string element in the vector to the console.
/// @param vec A vector of vectors of strings to be displayed in the console.
void printVector(vector<vector<string>> vec);
/// Helper for sorting alphabetically
bool mycomp(string a, string b);
/// @brief This function generates a vector containing all the files in a directory with the given filetype.
/// @param path Path to the directory, default : ./Graphs/X/
/// @param filetype The filetype, the function will return only this type. e.g. pdf. default:* (all filetypes)
/// @return A vector of strings containing all the filenames.
vector<string> glob(string path = "./Graphs/X/", string filetype = "*");
/// @brief Reads a vrp file and returns the values needed to generate the graph.
/// @param filename Path to the vrp file.
/// @return A tuple, containing a vector of a vector of three integers(x and y coordinate, demand) and an integer for the vehicle capacity.
tuple<vector<vector<int>>, int> readCVRP(string filename);

/// @brief This function will write the instance to a file
/// @param nodeCoordinatesAndDemand This is the vector conatining each nodes coordinate and demand i.e. (x,y,q)
/// @param capacity The maximum vehicle capacity
/// @param filePath Where to save the file
/// @param filename Filename (not used to save the file, only written in it)
void writeCVRP(vector<vector<int>> nodeCoordinatesAndDemand, int capacity, string filePath = "./Graphs/default.vrp", string filename = "CVRP_Graph_Instance");
/// @brief This function generates the position of the depot.
/// @param depotPosition String determining the depot position method, "C" center (500,500), "R" random "E" eccentric (0,0)
/// @return Vector with the x and y coordinates of the depot.
vector<int> depotPositionGenerator(string depotPosition);

/// @brief Divides all values of a vector by the div value (in place), now it is a cumulative distribution
/// @param vec Vector to be divided.
/// @param div Value to divide all elements by.
void double2dVectorDivisor(vector<vector<double>> &vec, double div);

/// @brief Asssigns based on the probability vector and a random value, doing binary search, the location of the value.
/// @param probVec Flattend vector with cummulative probabilities (i.e. last value = 1)
/// @param randomDouble Value to be found.
/// @return Returns two values, that will be the x and y value. (The x value is the flattend value divided by 1000 and the y is modulus 1000)
vector<int> findLocation(vector<double> &probVec, double randomDouble);

/// @brief Transforms a 2d vector to a 1 d.
/// @param orig Initial 2d vector
/// @return 1d flattened vector
vector<double> flatten(const vector<vector<double>> &orig);

/// @brief This function takes as parameter the probability distribution and an integer n. It then assigns
/// a location for n customers based on the given disstribution. It returns a vector of two integers (x, y) location of each customer.
/// @param probVec Distribution as in Uchoa et al. for the probability of each point receiving a customer.
/// @param n The number of customers to be assigned.
/// @return A vector of vectors each containing the x and y location for each customer. (n*2)
vector<vector<int>> generateCustomerClusters(vector<vector<double>> &probVec, int n);

/// @brief This function will generate the location for the cusstomers on the grid. It will do so for n customers and use the mode specified by the user.
/// @param n The number of customers that must be assigned in a graph.
/// @param customerPositiong The type of assignement "R" Random, "C" clustered "CR" half clustered & half random
/// @return A vector of vectors each containing the x and y location for each customer. (n*2)
vector<vector<int>> generateCustomerCoordinates(int n = 100, string customerPositiong = "R");

/// @brief This method generates the demand for each customer as described in Uchoa et al
/// @param coordinateVector  This is the coordinate vector for the customers
/// @param type This is the type of demand distribution (same order as in Uchoa et al)
/// @return A vector containing the demand for each customer (the 0th customer is the depot and has demand 0)
vector<int> generateDemand(vector<vector<int>> coordinateVector, int type);
/// @brief Generates a triangle distribution generator, with min mode and max.
/// @param min
/// @param peak
/// @param max
/// @return Double value from the triangular uniform distribution.
piecewise_linear_distribution<double> triangular_distribution(double min, double peak, double max);
/// @brief Generates the capacity for the vehicles, as in Uchoa et al
/// @param demandVector The demand values for each customer.
/// @return Integer corresponding to the maximum vehicle capacity for this instance.
int generateCapacity(vector<int> demandVector);
/// @brief This method generates an instance as descibed in Uchoa et al.
/// @param n The number of customers
/// @param depotPosition "C" center (500,500), "R" random "E" eccentric (0,0)
/// @param customerPositioning "R" Random, "C" clustered "CR" half clustered & half random
/// @param demandDistribution Integer for the six options in Uchoa et al
/// @return tuple with vector of vectors containing (x,y, demand) of each customer and an integer for the vehicle capacity
tuple<vector<vector<int>>, int> generateCVRPInstance(int n = 100, string depotPosition = "R", string customerPositioning = "R", int demandDistribution = 0);
   
/// @brief This function reads the solution file, and returns a vector of vectors. Each vector is a route, with customers in order.
/// @param filePath Path to the solution file
/// @return  The vector of vecotrs containing the routes and the cost of the solution.
tuple<vector<vector<int>>, int> readSolution(string filePath = "./Graphs/X/X-n101-k25.sol");

/// @brief This function saves the solution to a file
/// @param solutionVector Is the vecotr of vectors conatining the solution, each vector conatins the index order of the customers
/// @param cost The cost of the route
/// @param filePath The location to save the file
void writeSolution(vector<vector<int>> solutionVector, int cost, string filePath = "./Graphs/default.sol");

/// @brief This function takes the  customers  and calculates the distance between each vertex.
/// @param customers vector of vector of integeres, of the form (x,y, demand) for each vertes
/// @return vector of vector of integers of the distances. i.e. a symmetric matrix with the distances between the vertices
vector<vector<int>> calculateEdgeCost(vector<vector<int>> *customers);
/// @brief This function gets the binary matrix for the edges and returns the routes (implicitly starting and ending at 0).
/// @param edgeUsage vector of vector of integers, at position (i,j) 1 if verticess i and j are connected 0 otherwise.
/// @return A vector of vector of integers each subvector contains in order the indices of the verices visited.
vector<vector<int>> fromEdgeUsageToRouteSolution(vector<vector<int>> edgeUsage);

/// @brief This function takes the routes and the demands and ssums the demands for each route
/// @param routes A vector of vectors of integers, each contains a route. (the customers served)
/// @param demands The demand vector for each customer (depot included)
void printRouteAndDemand(vector<vector<int>> routes, vector<int> demands);

map<string, string> readJson(string filename = "cvrp.json");


#endif /* UTILITIES_G */