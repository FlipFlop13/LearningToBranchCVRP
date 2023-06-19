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
using namespace std;
namespace fs = filesystem;

/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of integer to be displayed in the console.
void printVector(vector<int> vec)
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
void printVector(vector<string> vec)
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
void printVector(vector<vector<int>> vec)
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
void printVector(vector<vector<string>> vec)
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
/// @brief This function generates a vector containing all the files in a directory with the given filetype.
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

/// @brief This function generates the position of the depot.
/// @param depotPosition String determining the depot position method, "C" center (500,500), "R" random "E" eccentric (0,0)
/// @return Vector with the x and y coordinates of the depot.
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

/// @brief Divides all values of a vector by the div value (in place), now it is a cumulative distribution
/// @param vec Vector to be divided.
/// @param div Value to divide all elements by.
void double2dVectorDivisor(vector<vector<double>> &vec, double div)
{
    int iMax = vec.size();
    int jMax = vec[0].size();
    for (int i = 0; i < iMax; i++)
    {
        for (int j = 0; j < jMax; j++)
        {
            vec[i][j] /= div;
        }
    }
}

/// @brief Asssigns based on the probability vector and a random value, doing binary search, the location of the value.
/// @param probVec Flattend vecotr with cummulative probabilities (i.e. last value = 1)
/// @param randomDouble Value to be found.
/// @return Returns two values, that will be the x and y value. (The x value is the flattend value divided by 1000 and the y is modulus 1000)
vector<int> findLocation(vector<double> &probVec, double randomDouble)
{
    int iMax = probVec.size();
    int iMin = 0;
    int iMiddle = iMin + (iMax - iMin) / 2;
    int idx = 0;
    while (true)
    {
        iMiddle = iMin + (iMax - iMin) / 2;
        if (probVec[iMiddle] > randomDouble)
        {
            iMax = iMiddle;
        }
        else
        {
            iMin = iMiddle;
        }
        if (probVec[iMiddle] > randomDouble && probVec[iMiddle - 1] < randomDouble)
        { // random value is in between these two points
            idx = iMiddle - 1;
            break;
        }
        else if (probVec[iMiddle] < randomDouble && probVec[iMiddle + 1] > randomDouble)
        {
            idx = iMiddle;
            break;
        }
    }
    vector<int> ret{idx / 1000, idx % 1000};
    return ret;
}

/// @brief Transforms a 2d vector to a 1 d.
/// @param orig Initial 2d vector
/// @return 1d flattened vector
vector<double> flatten(const vector<vector<double>> &orig)
{
    vector<double> ret;
    for (const auto &v : orig)
        ret.insert(ret.end(), v.begin(), v.end());
    return ret;
}

/// @brief This function takes as parameter the probability distribution and an integer n. It then assigns
/// a location for n customers based on the given disstribution. It returns a vector of two integers (x, y) location of each customer.
/// @param probVec Distribution as in Uchoa et al. for the probability of each point receiving a customer.
/// @param n The number of customers to be assigned.
/// @return A vector of vectors each containing the x and y location for each customer. (n*2)
vector<vector<int>> generateCustomerClusters(vector<vector<double>> &probVec, int n)
{
    vector<double> flatProb = flatten(probVec);
    vector<vector<int>> customerCoordinates(n, vector<int>(2));

    double lower_bound = 0;
    double upper_bound = 1;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;

    int iMax = probVec.size();
    int jMax = probVec[0].size();
    bool repeat = false;
    int idx = 0;
    while (idx < n)
    {
        // Getting a random double value
        double randomDouble = unif(re);
        vector<int> location = findLocation(flatProb, randomDouble);
        // if location already seen pick new number and go again
        repeat = false;
        for (vector<int> vec : customerCoordinates)
        {
            if (location[0] == vec[0] && location[1] == vec[1])
            {
                repeat = true;
                break;
            }
        }
        if (repeat)
        {
            continue;
        }
        // else (new location) idx++ and add location to set
        customerCoordinates[idx] = location;
        idx++;
    }
    return customerCoordinates;
}

/// @brief This function will generate the location for the cusstomers on the grid. It will do so for n customers and use the mode specified by the user.
/// @param n The number of customers that must be assigned in a graph.
/// @param customerPositiong The type of assignement "R" Random, "C" clustered "CR" half clustered & half random
/// @return A vector of vectors each containing the x and y location for each customer. (n*2)
vector<vector<int>> generateCustomerCoordinates(int n = 100, string customerPositiong = "R")
{
    if (customerPositiong == "R") // random method
    {
        vector<vector<int>> customerCoordinates(n, vector<int>(2));
        for (int i = 0; i < n; i++)
        {
            customerCoordinates[i][0] = rand() % 1000;
            customerCoordinates[i][1] = rand() % 1000;
        }
        return customerCoordinates;
    }
    else if (customerPositiong == "C") // clustered method
    {
        // int s = rand() % 6 + 3; // First get the number of clusters [3,8]
        uniform_int_distribution<int> unif(3, 8);
        default_random_engine re;
        int s = unif(re);
        vector<vector<int>> seedCoordinates(s, vector<int>(2)); // a vector containing the cluster seeds

        vector<vector<double>> probCoordinates(1000, vector<double>(1000)); // probability distributiion for customers

        cout << "Number of seeds:" << s << endl;
        uniform_int_distribution<int> unifD(1, 999);
        default_random_engine rd;
        for (int i = 0; i < s; i++) // create the location od the seed customers; They act as seed and as as customer hence two vectors;
        {
            seedCoordinates[i][0] = unifD(rd);
            seedCoordinates[i][1] = unifD(rd);
            cout << seedCoordinates[i][0] << " " << seedCoordinates[i][1] << endl;
        }

        double probSum = 0;
        for (int i = 0; i < 1000; i++) // calculate the probability of each point as in uchoa et al.
        {
            for (int j = 0; j < 1000; j++)
            {
                double probPoint = 0;
                for (vector<int> seed : seedCoordinates)
                {
                    if (seed[0] == i && seed[1] == j)
                    {                  // as all points are distinct the p of the seed location is 0
                        probPoint = 0; // set the p of the current point to 0
                        break;
                    }
                    double d = sqrt(pow((seed[0] - i), 2) + pow((seed[1] - j), 2));
                    double p = exp(-d / 40);
                    probPoint += p;
                }
                probSum += probPoint; // the probability of each point is the cumulative probaability, this allows binary search llater on
                probCoordinates[i][j] = probSum;
            }
        }
        double2dVectorDivisor(probCoordinates, probCoordinates[999][999]); // lets the cumulative probability be 1 at the last point
        vector<vector<int>> V = generateCustomerClusters(probCoordinates, (n - s));
        seedCoordinates.insert(seedCoordinates.end(), V.begin(), V.end()); // append to seed Coordinates, is in truth all coordinates now
        return seedCoordinates;
    }
    else
    { // Cluster and random method
        int half = n / 2;
        uniform_int_distribution<int> unif(3, 8);
        default_random_engine re;
        int s = unif(re);
        vector<vector<int>> customerCoordinates((half + s), vector<int>(2));
        cout << "Number of seeds:" << s << endl;

        // assign half randomly
        for (int i = 0; i < half; i++)
        {
            customerCoordinates[i][0] = rand() % 1000;
            customerCoordinates[i][1] = rand() % 1000;
        }

        vector<vector<int>> seedCoordinates(s, vector<int>(2));             // a vector containing the cluster seeds
        vector<vector<double>> probCoordinates(1000, vector<double>(1000)); // probability distributiion for customers

        uniform_int_distribution<int> unifD(1, 999);
        default_random_engine rd;
        for (int i = 0; i < s; i++) // create the location od the seed customers; They act as seed and as as customer hence two vectors;
        {
            seedCoordinates[i][0] = unifD(rd);
            seedCoordinates[i][1] = unifD(rd);
        }

        double probSum = 0;
        for (int i = 0; i < 1000; i++) // calculate the probability of each point as in uchoa et al.
        {
            for (int j = 0; j < 1000; j++)
            {
                double probPoint = 0;
                for (vector<int> seed : seedCoordinates)
                {
                    if (seed[0] == i && seed[1] == j)
                    {                  // as all points are distinct the p of the seed location is 0
                        probPoint = 0; // set the p of the current point to 0
                        break;
                    }
                    double d = sqrt(pow((seed[0] - i), 2) + pow((seed[1] - j), 2));
                    double p = exp(-d / 40);
                    probPoint += p;
                }
                probSum += probPoint; // the probability of each point is the cumulative probaability, this allows binary search llater on
                probCoordinates[i][j] = probSum;
            }
        }

        vector<vector<int>> V = generateCustomerClusters(probCoordinates, (n - s - half));
        customerCoordinates.insert(customerCoordinates.end(), V.begin(), V.end());
        return customerCoordinates;
    }
}

/// @brief This method generates the demand for each customer as described in Uchoa et al
/// @param coordinateVector  This is the coordinate vector for the customers
/// @param type This is the type of demand distribution (same order as in Uchoa et al)
/// @return A vector containing the demand for each customer (the 0th customer is the depot and has demand 0)
vector<int> generateDemand(vector<vector<int>> coordinateVector, int type)
{
    int n = coordinateVector.size();
    vector<int> demandVector(n);
    demandVector[0] = 0; // vector 0 is the depot and thus has demand 0
    switch (type)
    {
    case 0:
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1;
        }
        return demandVector;
    case 1:
    {
        int lower_bound = 1;
        int upper_bound = 10;
        uniform_int_distribution<int> unif(lower_bound, upper_bound);
        default_random_engine re;
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = unif(re);
        }
        return demandVector;
    }
    case 2:
    {

        uniform_int_distribution<int> unif(5, 10);
        default_random_engine re;
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = unif(re);
        }
        return demandVector;
    }
    case 3:
    {
        uniform_int_distribution<int> unif(1, 100);
        default_random_engine re;
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = unif(re);
        }
        return demandVector;
    }
    case 4:
    {
        uniform_int_distribution<int> unif(50, 100);
        default_random_engine re;
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = unif(re);
        }
        return demandVector;
    }
    case 5:
    {
        default_random_engine generator;

        uniform_int_distribution<int> smallDistr(1, 50);

        uniform_int_distribution<int> largeDistr(51, 100);

        uniform_int_distribution<int> variedDistr(1, 100);
        for (int i = 1; i < n; i++)
        {
            if (coordinateVector[i][0] > 500 && coordinateVector[i][1] > 500) // quadrant 1
            {
                demandVector[i] = largeDistr(generator);
            }
            else if (coordinateVector[i][0] < 500 && coordinateVector[i][1] > 500) // quadrant 2
            {
                demandVector[i] = smallDistr(generator);
            }
            else if (coordinateVector[i][0] < 500 && coordinateVector[i][1] < 500) // quadrant 3
            {
                demandVector[i] = largeDistr(generator);
            }
            else if (coordinateVector[i][0] > 500 && coordinateVector[i][1] < 500) // quadrant 4
            {
                demandVector[i] = smallDistr(generator);
            }
            else
            {
                demandVector[i] = variedDistr(generator);
            }
        }
        return demandVector;
    }
    case 6:
    {
        default_random_engine generator;

        uniform_real_distribution<double> thresholdDistr(0.7, 0.95);
        double thresholdVallue = thresholdDistr(generator);

        uniform_real_distribution<double> probDisstribution(0, 1);

        uniform_int_distribution<int> smallDistribution(1, 10);

        uniform_int_distribution<int> largeDistribution(50, 100);

        for (int i = 1; i < n; i++)
        {
            double probability = probDisstribution(generator);
            if (probability > thresholdVallue)
            { // with small probability get high demand
                demandVector[i] = largeDistribution(generator);
            }
            else
            { // with large probability get small demand
                demandVector[i] = smallDistribution(generator);
            }
        }
        return demandVector;
    }
    default:
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1;
        }
        return demandVector;
    }
}

/// @brief This method generates an instance as descibed in Uchoa et al.
/// @param n The number of customers
/// @param depotPosition "C" center (500,500), "R" random "E" eccentric (0,0)
/// @param customerPositioning "R" Random, "C" clustered "CR" half clustered & half random
/// @param demandDistribution Integer for the six options in Uchoa et al
void generate(int n = 100, string depotPosition = "R", string customerPositioning = "R", string demandDistribution = "U")
{
    vector<vector<int>> vecNCS(1, vector<int>(2));
    vecNCS[0] = depotPositionGenerator("R");

    customerPositioning = "C";
    vector<vector<int>> tempV = generateCustomerCoordinates(n, customerPositioning);
    vecNCS.insert(vecNCS.end(), tempV.begin(), tempV.end());
    vector<int> V1 = generateDemand(vecNCS, 6);

    // printIntVector2D(vecNCS)
}
