#include "./utilities.h"
/// This file contains all the helper functions, including custom printer functions and the graph creator functions
/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of integer to be displayed in the console.
void printVector(vector<float> vec)
{
    cout << endl
         << "Vector:" << endl;
    for (float i : vec)
    {
        cout << i << endl;
    }
}
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
/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of bool to be displayed in the console.
void printVector(vector<bool> vec)
{
    cout << endl
         << "Vector:" << endl;
    for (bool i : vec)
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
/// @brief This function prints every integer element in the vector to the console.
/// @param vec A vector of vectors of floats to be displayed in the console.
void printVector(vector<vector<float>> vec)
{
    cout << endl
         << "Vector:" << endl;
    for (vector<float> i : vec)
    {
        for (float j : i)
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
/// @brief This function prints every string element in the vector to the console.
/// @param vec A vector of vectors of bool to be displayed in the console.
void printVector(vector<vector<bool>> vec)
{
    cout << endl
         << "Vector:" << endl;
    for (vector<bool> i : vec)
    {
        for (bool j : i)
        {
            cout << j << " ";
        }
        cout << endl;
    }
}

/// Helper for sorting alphabetically
bool mycomp(string a, string b)
{
    return a < b;
}
/// @brief This function generates a vector containing all the files in a directory with the given filetype.
/// @param path Path to the directory, default : ./Graphs/X/
/// @param filetype The filetype, the function will return only this type. e.g. pdf. default:* (all filetypes)
/// @return A vector of strings containing all the filenames.
vector<string> glob(string path, string filetype)
{
    vector<string> filenames;
    int lenghtFileType = filetype.length();
    for (const auto &entry : filesystem::directory_iterator(path))
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

    sort(filenames.begin(), filenames.end(), mycomp);
    return filenames;
}
/// @brief Reads a vrp file and returns the values needed to generate the graph.
/// @param filename Path to the vrp file.
/// @return A tuple, containing a vector of a vector of three integers(x and y coordinate, demand) and an integer for the vehicle capacity.
tuple<vector<vector<int>>, int> readCVRP(string filename)
{
    vector<vector<int>> vecNCS(0, vector<int>(3));
    int capacity = -1;

    fstream newfile;
    newfile.open(filename, ios::in); // open a file to perform read operation using file object
    if (newfile.is_open())
    { // checking whether the file is open
        string tp;

        bool bcap = false, bncs = false, bds = false, bDepotS = false, bdim = false;
        int i = 0;

        while (getline(newfile, tp))
        { // read data from file object and put it into string. Loops for every row.
            int j = 0;
            stringstream ss(tp);
            string word;
            while (ss >> word)
            { // Extract word from the stream.
              // Check if we are in the section for node location, node demand or depot location
                if (word == "DIMENSION")
                {
                    bdim = true;
                    bcap = false;
                    bncs = false;
                    bds = false;
                    bDepotS = false;
                }
                if (word == "CAPACITY")
                {
                    bdim = false;
                    bcap = true;
                    bncs = false;
                    bds = false;
                    bDepotS = false;
                }
                if (word == "NODE_COORD_SECTION")
                {
                    bdim = false;
                    bcap = false;
                    bncs = true;
                    bds = false;
                    bDepotS = false;
                    i = -1;
                    continue;
                }
                if (word == "DEMAND_SECTION")
                {
                    bdim = false;
                    bcap = false;
                    bncs = false;
                    bds = true;
                    bDepotS = false;
                    i = -1;
                    continue;
                }
                if (word == "DEPOT_SECTION")
                {
                    bdim = false;
                    bcap = false;
                    bncs = false;
                    bds = false;
                    bDepotS = true;
                    i = -1;
                    continue;
                }
                if (word == "EOF")
                {
                    tuple<vector<vector<int>>, int> retTuple(vecNCS, capacity);
                    // close the file object.
                    newfile.close();
                    return retTuple;
                }

                if (bdim)
                {
                    try
                    {
                        int dimension = stoi(word);
                        vector<vector<int>> t(dimension, vector<int>(3));
                        vecNCS.insert(vecNCS.end(), t.begin(), t.end());
                    }
                    catch (invalid_argument)
                    {
                        continue;
                    }
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
                }
                if (bncs)
                {
                    if (j == 0)
                    {
                        j++; // in the files the first value is the index, then X then Y.
                        continue;
                    }
                    int strToInt;
                    strToInt = stoi(word);

                    vecNCS[i][(j - 1)] = strToInt;
                }
                if (bds)
                {
                    if (j == 0)
                    {
                        j++; // in the files the first value is the index, then the demand.
                        continue;
                    }
                    int strToInt;
                    strToInt = stoi(word);
                    vecNCS[i][2] = strToInt;
                }
                j++;
            }
            i++;
        }

        cout << endl;
        // close the file object.
        newfile.close();
    }
    tuple<vector<vector<int>>, int> retTuple(vecNCS, capacity);
    return retTuple;
}
/// @brief This function will write the instance to a file
/// @param nodeCoordinatesAndDemand This is the vector conatining each nodes coordinate and demand i.e. (x,y,q)
/// @param capacity The maximum vehicle capacity
/// @param filename Where to save the file
/// @param name Filename (not used to save the file, only written in it)
void writeCVRP(vector<vector<int>> nodeCoordinatesAndDemand, int capacity, string filename, string name)
{
    ofstream fw(filename, ofstream::out);
    if (!fw.is_open())
    {
        return;
    }
    int rowNumber = nodeCoordinatesAndDemand.size();
    fw << "NAME:        " << name << "\n";
    fw << "COMMENT : 	\"Generated by Salomons, Philip (2023)\""
       << "\n";
    fw << "TYPE : 	CVRP"
       << "\n";
    fw << "DIMENSION : 	" << rowNumber << "\n";
    fw << "EDGE_WEIGHT_TYPE : 	EUC_2D	"
       << "\n";
    fw << "CAPACITY : 	" << capacity << "\n";
    fw << "NODE_COORD_SECTION"
       << "\n";

    for (int i = 0; i < rowNumber; i++)
    { // in the vrp files, the nodes are 1 based...

        fw << (i + 1) << "    " << nodeCoordinatesAndDemand[i][0] << "    " << nodeCoordinatesAndDemand[i][1] << "\n";
    }
    fw << "DEMAND_SECTION"
       << "\n";

    for (int i = 0; i < rowNumber; i++)
    { // in the vrp files, the nodes are 1 based...
        fw << (i + 1) << "    " << nodeCoordinatesAndDemand[i][2] << "\n";
    }

    fw << "DEPOT_SECTION"
       << "\n";
    fw << 1 << "\n";
    fw << -1 << "\n";
    fw << "EOF"
       << "\n";
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
/// @param probVec Flattend vector with cummulative probabilities (i.e. last value = 1)
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

    int iMax = probVec.size();
    int jMax = probVec[0].size();
    bool repeat = false;
    int idx = 0;
    while (idx < n)
    {
        // Getting a random double value
        double randomDouble = (float)(rand() % 100000) / 100000;
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
        // else (new location)  add location to set and idx++
        customerCoordinates[idx] = location;
        idx++;
    }

    return customerCoordinates;
}

/// @brief This function will generate the location for the cusstomers on the grid. It will do so for n customers and use the mode specified by the user.
/// @param n The number of customers that must be assigned in a graph.
/// @param customerPositiong The type of assignement "R" Random, "C" clustered "CR" half clustered & half random
/// @return A vector of vectors each containing the x and y location for each customer. (n*2)
vector<vector<int>> generateCustomerCoordinates(int n, string customerPositiong)
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
        // First get the number of clusters [3,8]
        int s = (rand() % 6) + 3;                               // This generates a random number between 3 and 8
        vector<vector<int>> seedCoordinates(s, vector<int>(2)); // a vector containing the cluster seeds

        vector<vector<double>> probCoordinates(1000, vector<double>(1000)); // probability distributiion for customers

        for (int i = 0; i < s; i++) // create the location of the seed customers; They act as seed and as as customer hence two vectors;
        {
            seedCoordinates[i][0] = rand() % 1000;
            seedCoordinates[i][1] = rand() % 1000;
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
        if ((n - s) > 0)
        { // There are more customers needed than there are seeds
            vector<vector<int>> V = generateCustomerClusters(probCoordinates, (n - s));
            seedCoordinates.insert(seedCoordinates.end(), V.begin(), V.end()); // append to seed Coordinates, is in truth all coordinates now
            return seedCoordinates;
        }
        else if ((n - s) == 0)
        { // THe number of coordinates needed is the same as the number of seeds
            return seedCoordinates;
        }
        else
        { // There are more seeds tha customers needed, return the first n seeds
            vector<vector<int>> v;
            v = vector<vector<int>>(seedCoordinates.begin(), seedCoordinates.begin() + n);
            return v;
        }
    }
    else
    {                             // Cluster and random method
        int s = (rand() % 6) + 3; // This generates a random number between 3 and 8 for the number of clusters
        int half = n / 2;
        vector<vector<int>> customerCoordinates((half + s), vector<int>(2));

        // assign half randomly
        for (int i = 0; i < half; i++)
        {
            customerCoordinates[i][0] = rand() % 1000;
            customerCoordinates[i][1] = rand() % 1000;
        }

        vector<vector<int>> seedCoordinates(s, vector<int>(2));             // a vector containing the cluster seeds
        vector<vector<double>> probCoordinates(1000, vector<double>(1000)); // probability distributiion for customers
        for (int i = 0; i < s; i++)                                         // create the location of the seed customers; They act as seed and as as customer hence two vectors;
        {
            seedCoordinates[i][0] = rand() % 1000;
            seedCoordinates[i][1] = rand() % 1000;
            customerCoordinates[half + i][0] = seedCoordinates[i][0];
            customerCoordinates[half + i][1] = seedCoordinates[i][1];
        }

        double probSum = 0;
        for (int i = 0; i < 1000; i++) // calculate the probability of each point as in uchoa et al.
        {
            for (int j = 0; j < 1000; j++)
            {
                double probPoint = 0;
                for (vector<int> seed : seedCoordinates) // Add the weights to each point based on the distance to each seed
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
                probSum += probPoint; // the probability of each point is the cumulative probability, this allows binary search later on
                probCoordinates[i][j] = probSum;
            }
        }

        double2dVectorDivisor(probCoordinates, probCoordinates.back().back());

        if (n > (s + half))
        {
            vector<vector<int>> V = generateCustomerClusters(probCoordinates, (n - (s + half)));
            customerCoordinates.insert(customerCoordinates.end(), V.begin(), V.end());

            return customerCoordinates;
        }
        else
        {
            vector<vector<int>> V;
            V = vector<vector<int>>(customerCoordinates.begin(), customerCoordinates.begin() + n);

            return V;
        }
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
    case 0: // unitary demand
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1;
        }
        return demandVector;
    }
    case 1: // Uniformly distributed [1,10]
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1 + rand() % 10;
        }
        return demandVector;
    }
    case 2: // Uniformly distributed [5,10]
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 5 + rand() % 5;
        }
        return demandVector;
    }
    case 3: // Uniformly distributed [1,100]
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1 + rand() % 100;
        }
        return demandVector;
    }
    case 4: // Uniformly distributed [50,100]
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 50 + rand() % 51;
        }
        return demandVector;
    }
    case 5: // quadrant dependent
    {
        for (int i = 1; i < n; i++)
        {
            if (coordinateVector[i][0] > 500 && coordinateVector[i][1] > 500) // quadrant 1
            {
                demandVector[i] = 51 + rand() % 50; // Large quadrant
            }
            else if (coordinateVector[i][0] < 500 && coordinateVector[i][1] > 500) // quadrant 2
            {
                demandVector[i] = 1 + rand() % 50; // Small quadrant
            }
            else if (coordinateVector[i][0] < 500 && coordinateVector[i][1] < 500) // quadrant 3
            {
                demandVector[i] = 51 + rand() % 50; // Large quadrant
            }
            else if (coordinateVector[i][0] > 500 && coordinateVector[i][1] < 500) // quadrant 4
            {
                demandVector[i] = 1 + rand() % 50; // Small quadrant
            }
            else
            {
                demandVector[i] = 1 + rand() % 100; // On the quadrand edge (whole disrtibution)
            }
        }
        return demandVector;
    }
    case 6: // many small values few large values
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
    {
        for (int i = 1; i < n; i++)
        {
            demandVector[i] = 1;
        }
        return demandVector;
    }
    }
}
/// @brief Generates a triangle distribution generator, with min mode and max.
/// @param min
/// @param peak
/// @param max
/// @return Double value from the triangular uniform distribution.
piecewise_linear_distribution<double> triangular_distribution(double min, double peak, double max)
{
    std::array<double, 3> i{min, peak, max};
    std::array<double, 3> w{0, 1, 0};
    return std::piecewise_linear_distribution<double>{i.begin(), i.end(), w.begin()};
}
/// @brief Generates the capacity for the vehicles, as in Uchoa et al
/// @param demandVector The demand values for each customer.
/// @return Integer corresponding to the maximum vehicle capacity for this instance.
int generateCapacity(vector<int> demandVector)
{
    int n = demandVector.size() - 1; // The depot does not count here
    int totalDemand = 0;
    for (int i : demandVector)
    {
        totalDemand += i;
    }
    // vector<int> pdf_triangular{2,8 ,14,20,19,18,17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    vector<int> cdf_triangular{2, 10, 24, 44, 63, 81, 98, 114, 129, 143, 156, 168, 179, 189, 198, 206, 213, 219, 224, 228, 231, 233, 234};
    int rs = rand() % 234;
    int r = 0;
    for (int i = 0; i < 23; i++)
    {
        if (i < cdf_triangular[i])
        {
            r = i;
            break;
        }
    }

    r = r + 3;
    cout << "R: " << r << endl;
    int Q = ceil((r * totalDemand) / n);
    return Q;
}

/// @brief This method generates an instance as descibed in Uchoa et al.
/// @param n The number of customers
/// @param depotPosition "C" center (500,500), "R" random "E" eccentric (0,0)
/// @param customerPositioning "R" Random, "C" clustered "CR" half clustered & half random
/// @param demandDistribution Integer for the six options in Uchoa et al
/// @return tuple with vector of vectors containing (x,y, demand) of each customer and an integer for the vehicle capacity
tuple<vector<vector<int>>, int> generateCVRPInstance(int n, string depotPosition, string customerPositioning, int demandDistribution)
{
    vector<vector<int>> coordinates(1, vector<int>(2));
    coordinates[0] = depotPositionGenerator(depotPosition);

    vector<vector<int>> customerCoordinates = generateCustomerCoordinates(n, customerPositioning);
    coordinates.insert(coordinates.end(), customerCoordinates.begin(), customerCoordinates.end());

    vector<int> demandVector = generateDemand(coordinates, demandDistribution);

    vector<vector<int>> returnVector((n + 1), vector<int>(3));

    for (int i = 0; i <= n; i++)
    {
        returnVector[i][0] = coordinates[i][0];
        returnVector[i][1] = coordinates[i][1];
        returnVector[i][2] = demandVector[i];
    }
    returnVector[0][2] = 0; // depot demand is 0

    int capacity = generateCapacity(demandVector);
    tuple<vector<vector<int>>, int> retTuple(returnVector, capacity);
    return retTuple;
}
/// @brief This function reads the solution file, and returns a vector of vectors. Each vector is a route, with customers in order.
/// @param filePath Path to the solution file
/// @return  The vector of vecotrs containing the routes and the cost of the solution.
tuple<vector<vector<int>>, int> readSolution(string filePath)
{
    fstream newfile;
    newfile.open(filePath, ios::in); // open a file to perform read operation using file object
    if (!newfile.is_open())
    {
        exit(-1);
    }
    vector<vector<int>> solutionVector;
    int cost = -1;
    string tp;
    bool lastRow = false; // it reaches the last row when 'Cost' is seen
    while (getline(newfile, tp))
    { // read data from file object and put it into string. Loops for every row.
        int j = 0;
        stringstream ss(tp);
        string word;
        vector<int> routeVector;
        while (ss >> word)
        {
            if (word == "Cost")
            {
                lastRow = true;
            }
            if (lastRow) // now we get the cost
            {
                try
                {
                    cost = stoi(word);
                    tuple<vector<vector<int>>, int> returnTuple(solutionVector, cost);
                    return returnTuple;
                }
                catch (invalid_argument)
                {
                    continue;
                }
            }
            try
            {
                int nextCostumer = stoi(word);
                routeVector.push_back(nextCostumer);
            }
            catch (invalid_argument)
            {
                continue;
            }
        }
        solutionVector.push_back(routeVector);
    }
    tuple<vector<vector<int>>, int> returnTuple(solutionVector, cost);
    return returnTuple;
}

/// @brief This function saves the solution to a file
/// @param solutionVector Is the vecotr of vectors conatining the solution, each vector conatins the index order of the customers
/// @param cost The cost of the route
/// @param filePath The location to save the file
void writeSolution(vector<vector<int>> solutionVector, int cost, string filePath)
{
    ofstream fw(filePath, ofstream::out);
    if (!fw.is_open())
    {
        return;
    }
    int i = 0;
    for (vector<int> route : solutionVector)
    {
        i++; // route numbers in the solution are 1 based
        fw << "Route #" << i << ": ";
        for (int index : route)
        {
            fw << index << " ";
        }
        fw << "\n";
    }
    fw << "Cost: " << cost << "\n";
}
/// @brief This function takes the  customers  and calculates the distance between each vertex.
/// @param customers vector of vector of integeres, of the form (x,y, demand) for each vertes
/// @return vector of vector of integers of the distances. i.e. a symmetric matrix with the distances between the vertices
vector<vector<int>> calculateEdgeCost(vector<vector<int>> *customers)
{
    int n = customers->size();
    vector<vector<int>> edgeCost(n, vector<int>(n, 10000)); // Set all values to be a very large number
    for (int i = 0; i < n; i++)
    {
        for (int j = (i + 1); j < n; j++)
        {
            int x0 = customers->at(i).at(0);
            int x1 = customers->at(j).at(0);
            int y0 = customers->at(i).at(1);
            int y1 = customers->at(j).at(1);
            int eucDistance = sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2));

            edgeCost.at(i).at(j) = eucDistance;
            edgeCost.at(j).at(i) = eucDistance;
        }
    }
    return edgeCost;
}

/// @brief This function gets the binary matrix for the edges and returns the routes (implicitly starting and ending at 0).
/// @param edgeUsage vector of vector of integers, at position (i,j) 1 if verticess i and j are connected 0 otherwise.
/// @return A vector of vector of integers each subvector contains in order the indices of the verices visited.
vector<vector<int>> fromEdgeUsageToRouteSolution(vector<vector<int>> edgeUsage)
{
    int n = edgeUsage.size();
    vector<vector<int>> solutionVector;
    int lastVertex = 0;

    for (int i = 0; i < n; i++)
    {
        vector<int> route;
        if (edgeUsage.at(i).at(0) == 0)
        {
            continue; // no route starts here
        }
        else if (edgeUsage.at(i).at(0) == 2)
        {
            route.push_back(i); // Route consist only of one vertex
            edgeUsage.at(i).at(0) = 0;
            edgeUsage.at(0).at(i) = 0;
        }
        else
        { // edgeUsage must be one
            edgeUsage.at(i).at(0) = 0;
            edgeUsage.at(0).at(i) = 0;
            route.push_back(i);
            lastVertex = i;
            int j = 0; // I think it can be one, for later

            while (true)
            {   
                if (edgeUsage.at(lastVertex).at(j) == 1)
                {
                    if (j == 0)
                    { // the route has finished
                        edgeUsage.at(lastVertex).at(0) = 0;
                        edgeUsage.at(0).at(lastVertex) = 0;
                        break;
                    }
                    // else add the vertex to the route, and start looking where it connects to
                    route.push_back(j);
                    edgeUsage.at(j).at(lastVertex) = 0;
                    edgeUsage.at(lastVertex).at(j) = 0;
                    lastVertex = j;
                    j = 0;
                    continue;
                }
                j++;
            }
        }
        solutionVector.push_back(route);
    }
    printVector(solutionVector);
    return solutionVector;
}

/// @brief This function takes the routes and the demands and ssums the demands for each route
/// @param routes A vector of vectors of integers, each contains a route. (the customers served)
/// @param demands The demand vector for each customer (depot included)
void printRouteAndDemand(vector<vector<int>> routes, vector<int> demands)
{

    for (vector<int> route : routes)
    {
        cout << "Route: ";
        int demandSum = 0;
        for (int costumer : route)
        {
            cout << costumer << " ";
            demandSum += demands[costumer];
        }
        cout << "Demand sum: " << demandSum << endl;
    }
}

map<string, string> readJson(string filename)
{

    fstream newfile;
    newfile.open(filename, ios::in); // open a file to perform read operation using file object
    if (!newfile.is_open())
    {
        exit(-1);
    }
    map<string, string> map;
    string tp;
    string type = "key";
    string key;
    string term;
    while (getline(newfile, tp))
    { // read data from file object and put it into string. Loops for every row.
        int j = 0;
        stringstream ss(tp);
        string word;
        vector<int> routeVector;
        while (ss >> word)
        {
            if (word[0] == '/' && word[1] == '/')
            { // is a comment
                break;
            }
            for (int i = 0; i < word.size(); i++)
            {
                if (word[i] == '"')
                {
                    continue;
                }
                else if (word[i] == ':')
                {
                    type = ":";
                    continue;
                }
                else if (word[i] == ',')
                {
                    continue;
                }
                term += word[i];
            }
            if (type == "key")
            {
                key = term;
                term = "";
            }
            else if (type == "term")
            {
                map[key] = term;
                cout << key << "<->" << term << endl;
                key = "";
                term = "";
                type = "key";
            }
            else if (type == ":")
            {
                type = "term";
            }
        }
    }
    return map;
}

/// @brief Creates a vector of dimension 4
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param w Size of dimension 4 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<vector<float>>>> fourDimensionVectorCreator(int x, int y, int z, int w, float defaultValue)
{
    vector<vector<vector<vector<float>>>> bigBoy(x, vector<vector<vector<float>>>(y, vector<vector<float>>(z, vector<float>(w, defaultValue))));
    return bigBoy;
}
/// @brief Creates a vector of dimension 4
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param w Size of dimension 4 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<vector<int>>>> fourDimensionVectorCreator(int x, int y, int z, int w, int defaultValue)
{
    vector<vector<vector<vector<int>>>> bigBoy(x, vector<vector<vector<int>>>(y, vector<vector<int>>(z, vector<int>(w, defaultValue))));
    return bigBoy;
}
/// @brief Creates a vector of dimension 4
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param w Size of dimension 4 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<vector<bool>>>> fourDimensionVectorCreator(int x, int y, int z, int w, bool defaultValue)
{
    vector<vector<vector<vector<bool>>>> bigBoy(x, vector<vector<vector<bool>>>(y, vector<vector<bool>>(z, vector<bool>(w, defaultValue))));
    return bigBoy;
}

/// @brief Creates a vector of dimension 3
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<float>>> threeDimensionVectorCreator(int x, int y, int z, float defaultValue)
{
    vector<vector<vector<float>>> bigBoy(x, vector<vector<float>>(y, vector<float>(z, defaultValue)));
    return bigBoy;
}
/// @brief Creates a vector of dimension 3
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<int>>> threeDimensionVectorCreator(int x, int y, int z, int defaultValue)
{
    vector<vector<vector<int>>> bigBoy(x, vector<vector<int>>(y, vector<int>(z, defaultValue)));
    return bigBoy;
}
/// @brief Creates a vector of dimension 3
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param z Size of dimension 3 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<vector<bool>>> threeDimensionVectorCreator(int x, int y, int z, bool defaultValue)
{
    vector<vector<vector<bool>>> bigBoy(x, vector<vector<bool>>(y, vector<bool>(z, defaultValue)));
    return bigBoy;
}

/// @brief Creates a vector of dimension 2
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<float>> twoDimensionVectorCreator(int x, int y, float defaultValue)
{
    vector<vector<float>> bigBoy(x, vector<float>(y, defaultValue));
    return bigBoy;
}
/// @brief Creates a vector of dimension 2
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<int>> twoDimensionVectorCreator(int x, int y, int defaultValue)
{
    vector<vector<int>> bigBoy(x, vector<int>(y, defaultValue));
    return bigBoy;
}
/// @brief Creates a vector of dimension 2
/// @param x Size of dimension 1
/// @param y Size of dimension 2 
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<vector<bool>> twoDimensionVectorCreator(int x, int y, bool defaultValue)
{
    vector<vector<bool>> bigBoy(x, vector<bool>(y, defaultValue));
    return bigBoy;
}

/// @brief Creates a vector of dimension 1
/// @param x Size of dimension 1
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<float> oneDimensionVectorCreator(int x, float defaultValue)
{
    vector<float> bigBoy(x, defaultValue);
    return bigBoy;
}
/// @brief Creates a vector of dimension 1
/// @param x Size of dimension 1
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<int> oneDimensionVectorCreator(int x, int defaultValue)
{
    vector<int> bigBoy(x, defaultValue);
    return bigBoy;
}
/// @brief Creates a vector of dimension 1
/// @param x Size of dimension 1
/// @param defaultValue Value used to fill the vector
/// @return Vector 
vector<bool> oneDimensionVectorCreator(int x, bool defaultValue)
{
    vector<bool> bigBoy(x, defaultValue);
    return bigBoy;
}

/// @brief Calculates the likeness between the two vectors
/// @param v0 Vector of weights of an individual
/// @param v1 Vector of weights of another individual
/// @return tuple of the weight likeness and the active connections likeness
tuple<float, int> fourDimensionLikeness(vector<vector<vector<vector<float>>>> &v0, vector<vector<vector<vector<float>>>> &v1)
{
    int ED = 0;
    float W = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();
    int s2 = v0[0][0].size();
    int s3 = v0[0][0][0].size();
    int i, j, k, l;

#pragma omp parallel for private(i, j, k, l) shared(v0, v1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            for (k = 0; k < s2; ++k)
            {
                for (l = 0; l < s3; ++l)
                {
                    if (abs(v0[i][j][k][l]) < EPSU && abs(v1[i][j][k][l]) < EPSU) // If they are both zero they are both innexistent (skip)
                    {
                        continue;
                    }
                    else if (!(abs(v0[i][j][k][l]) < EPSU) && !(abs(v1[i][j][k][l]) < EPSU)) // If they are both not zero they  both exist
                    {
                        W += abs(v0[i][j][k][l] - v1[i][j][k][l]);
                    }
                    else // Else one exist and the other doesnt add ED counter
                    {
                        ED++;
                    }
                }
            }
        }
    }
    return make_tuple(W, ED);
}
/// @brief Calculates the likeness between the two vectors
/// @param v0 Vector of weights of an individual
/// @param v1 Vector of weights of another individual
/// @return tuple of the weight likeness and the active connections likeness
tuple<float, int> threeDimensionLikeness(vector<vector<vector<float>>> &v0, vector<vector<vector<float>>> &v1)
{
    int ED = 0;
    float W = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();
    int s2 = v0[0][0].size();
    int i, j, k;

#pragma omp parallel for private(i, j, k) shared(v0, v1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            for (k = 0; k < s2; ++k)
            {
                if (abs(v0[i][j][k]) < EPSU && abs(v1[i][j][k]) < EPSU) // If they are both zero they are both innexistent (skip)
                {
                    continue;
                }
                else if (!(abs(v0[i][j][k]) < EPSU) && !(abs(v1[i][j][k]) < EPSU)) // If they are both not zero they  both exist
                {
                    W += abs(v0[i][j][k] - v1[i][j][k]);
                }
                else // Else one exist and the other doesnt add ED counter
                {
                    ED++;
                }
            }
        }
    }
    return make_tuple(W, ED);
}
/// @brief Calculates the likeness between the two vectors
/// @param v0 Vector of weights of an individual
/// @param v1 Vector of weights of another individual
/// @return tuple of the weight likeness and the active connections likeness
tuple<float, int> twoDimensionLikeness(vector<vector<float>> &v0, vector<vector<float>> &v1)
{
    int ED = 0;
    float W = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();
    int i, j;

#pragma omp parallel for private(i, j) shared(v0, v1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {

            if (abs(v0[i][j]) < EPSU && abs(v1[i][j]) < EPSU) // If they are both zero they are both innexistent (skip)
            {
                continue;
            }
            else if (!(abs(v0[i][j]) < EPSU) && !(abs(v1[i][j]) < EPSU)) // If they are both not zero they  both exist
            {
                W += abs(v0[i][j] - v1[i][j]);
            }
            else // Else one exist and the other doesnt add ED counter
            {
                ED++;
            }
        }
    }
    return make_tuple(W, ED);
}
/// @brief Calculates the likeness between the two vectors
/// @param v0 Vector of weights of an individual
/// @param v1 Vector of weights of another individual
/// @return tuple of the weight likeness and the active connections likeness
tuple<float, int> oneDimensionLikeness(vector<float> &v0, vector<float> &v1)
{
    int ED = 0;
    float W = 0;
    int s0 = v0.size();
    int i;

#pragma omp parallel for private(i) shared(v0, v1)
    for (i = 0; i < s0; ++i)
    {
            if (abs(v0[i]) < EPSU && abs(v1[i]) < EPSU) // If they are both zero they are both innexistent (skip)
            {
                continue;
            }
            else if (!(abs(v0[i]) < EPSU) && !(abs(v1[i]) < EPSU)) // If they are both not zero they  both exist
            {
                W += abs(v0[i] - v1[i]);
            }
            else // Else one exist and the other doesnt add ED counter
            {
                ED++;
            }
        
    }
    return make_tuple(W, ED);
}

/// @brief Counts the number of active boolean values in a vector
/// @param v0 Vector of boolean values
/// @return  Number of active boolean values (Number of 1/true)
int boolVectorCounter(vector<vector<vector<vector<bool>>>> &v0)
{
    int count = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();
    int s2 = v0[0][0].size();
    int s3 = v0[0][0][0].size();
    int i, j, k;

#pragma omp parallel for private(i, j, k) shared(v0, count)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            for (k = 0; k < s2; ++k)
            {
                auto countT = std::count(v0[i][j][k].begin(), v0[i][j][k].end(), true);
                count += countT;
            }
        }
    }
    return count;
}
/// @brief Counts the number of active boolean values in a vector
/// @param v0 Vector of boolean values
/// @return  Number of active boolean values (Number of 1/true)
int boolVectorCounter(vector<vector<vector<bool>>> &v0)
{
    int count = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();
    int s2 = v0[0][0].size();
    int i, j, k, l;

#pragma omp parallel for private(i, j) shared(v0, count)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {

            auto countT = std::count(v0[i][j].begin(), v0[i][j].end(), true);
            count += countT;
        }
    }
    return count;
}
/// @brief Counts the number of active boolean values in a vector
/// @param v0 Vector of boolean values
/// @return  Number of active boolean values (Number of 1/true)
int boolVectorCounter(vector<vector<bool>> &v0)
{
    int count = 0;
    int s0 = v0.size();
    int s1 = v0[0].size();

    int i;

#pragma omp parallel for private(i, j, k, l) shared(v0, count)
    for (i = 0; i < s0; ++i)
    {

        auto countT = std::count(v0[i].begin(), v0[i].end(), true);
        count += countT;
    }
    return count;
}

/// @brief This function will take the activation layers of the two parents and of the kid, and will choose, for genes that both parents have, a value from eihter parent at random
/// @param vF0 Weight layer of parent 0
/// @param vF1 Weight laer of parent 1
/// @param vFkid Weight layer of the kid (This will be updated here)
/// @param vA0  Activation layer of parent 0
/// @param vA1 Activation layer of parent 1
/// @param vAkid Activation layer of kid 
void chooseRandomForMatchingGenes(vector<vector<vector<vector<float>>>> &vF0, vector<vector<vector<vector<float>>>> &vF1, vector<vector<vector<vector<float>>>> &vFkid, vector<vector<vector<vector<bool>>>> &vA0, vector<vector<vector<vector<bool>>>> &vA1, vector<vector<vector<vector<bool>>>> &vAkid)
{
    int i, j, k, l;
    int s0 = vF0.size();
    int s1 = vF0[0].size();
    int s2 = vF0[0][0].size();
    int s3 = vF0[0][0][0].size();
    float randomFloat;
#pragma omp parallel for private(i, j, k, l) shared(vFkid, vF0, vF1, vAkid, vA0, vA1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            for (k = 0; k < s2; ++k)
            {
                for (l = 0; l < s3; ++l)
                {
                    if (!(abs(vF0[i][j][k][l]) < EPSU) && !(abs(vF1[i][j][k][l]) < EPSU)) // If they are both not zero they  both exist, and thus matching
                    {

                        randomFloat = ((float)rand() / (float)RAND_MAX);
                        if (randomFloat > 0.5)
                        {
                            vFkid[i][j][k][l] = vF0[i][j][k][l];
                            vAkid[i][j][k][l] = vA0[i][j][k][l];
                        }
                        else
                        {
                            vFkid[i][j][k][l] = vF1[i][j][k][l];
                            vAkid[i][j][k][l] = vA1[i][j][k][l];
                        }
                    }
                }
            }
        }
    }
}
/// @brief This function will take the activation layers of the two parents and of the kid, and will choose, for genes that both parents have, a value from eihter parent at random
/// @param vF0 Weight layer of parent 0
/// @param vF1 Weight laer of parent 1
/// @param vFkid Weight layer of the kid (This will be updated here)
/// @param vA0  Activation layer of parent 0
/// @param vA1 Activation layer of parent 1
/// @param vAkid Activation layer of kid
void chooseRandomForMatchingGenes(vector<vector<vector<float>>> &vF0, vector<vector<vector<float>>> &vF1, vector<vector<vector<float>>> &vFkid, vector<vector<vector<bool>>> &vA0, vector<vector<vector<bool>>> &vA1, vector<vector<vector<bool>>> &vAkid)
{

    int i, j, k;

    int s0 = vF0.size();
    int s1 = vF0[0].size();
    int s2 = vF0[0][0].size();
    float randomFloat;
#pragma omp parallel for private(i, j, k) shared(vFkid, vF0, vF1, vAkid, vA0, vA1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            for (k = 0; k < s2; ++k)
            {
                if (!(abs(vF0[i][j][k]) < EPSU) && !(abs(vF1[i][j][k]) < EPSU)) // If they are both not zero they  both exist, and thus matching
                {
                    randomFloat = ((float)rand() / (float)RAND_MAX);
                    if (randomFloat > 0.5)
                    {
                        vFkid[i][j][k] = vF0[i][j][k];
                        vAkid[i][j][k] = vA0[i][j][k];
                    }
                    else
                    {
                        vFkid[i][j][k] = vF1[i][j][k];
                        vAkid[i][j][k] = vA1[i][j][k];
                    }
                }
            }
        }
    }
}
/// @brief This function will take the activation layers of the two parents and of the kid, and will choose, for genes that both parents have, a value from eihter parent at random
/// @param vF0 Weight layer of parent 0
/// @param vF1 Weight laer of parent 1
/// @param vFkid Weight layer of the kid (This will be updated here)
/// @param vA0  Activation layer of parent 0
/// @param vA1 Activation layer of parent 1
/// @param vAkid Activation layer of kid 
void chooseRandomForMatchingGenes(vector<vector<float>> &vF0, vector<vector<float>> &vF1, vector<vector<float>> &vFkid, vector<vector<bool>> &vA0, vector<vector<bool>> &vA1, vector<vector<bool>> &vAkid)
{

    int i, j;

    int s0 = vF0.size();
    int s1 = vF0[0].size();
    float randomFloat;
#pragma omp parallel for private(i, j) shared(vFkid, vF0, vF1, vAkid, vA0, vA1)
    for (i = 0; i < s0; ++i)
    {
        for (j = 0; j < s1; ++j)
        {
            if (!(abs(vF0[i][j]) < EPSU) && !(abs(vF1[i][j]) < EPSU)) // If they are both not zero they  both exist, and thus matching
            {
                randomFloat = ((float)rand() / (float)RAND_MAX);
                if (randomFloat > 0.5)
                {
                    vFkid[i][j] = vF0[i][j];
                    vAkid[i][j] = vA0[i][j];
                }
                else
                {
                    vFkid[i][j] = vF1[i][j];
                    vAkid[i][j] = vA1[i][j];
                }
            }
        }
    }
}

/// @brief Get all the permutations of the CVRP modes we can generate
/// @return a vector filled with tuples that can be used to generate an instance
vector<tuple<string, string, int>> getCVRPPermutations()
{
    vector<tuple<string, string, int>> permutations;
    vector<string> depotLocations{"E", "C", "R"};
    vector<string> customerDistributions{"R", "C", "CR"};
    for (auto x : depotLocations)
    {
        for (auto y : customerDistributions)
        {
            for (int i = 0; i < 6; ++i)
            {
                permutations.push_back(make_tuple(x, y, i));
            }
        }
    }
    return permutations;
}


/// @brief Tries to clear the memory of these vectors
/// @param vec A vector to be forgotten
void clearAndResizeVector(vector<vector<vector<vector<bool>>>> *vec)
{
    int s1, s2, s3;
    int i, j, k;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        s2 = vec->at(i).size();
        for (j = 0; j < s2; ++j)
        {
            s3 = vec->at(i).at(j).size();

            for (k = 0; k < s3; ++k)
            {
                // vec->at(i).at(j).at(k).clear();
                vec->at(i).at(j).at(k).resize(0);
                vec->at(i).at(j).at(k).shrink_to_fit();
                // vec->at(i).at(j).at(k).~vector();
                // vector<float>().swap(vec->at(i).at(j).at(k));
            }
            // vec->at(i).at(j).clear();
            vec->at(i).at(j).resize(0);
            vec->at(i).at(j).shrink_to_fit();
        }
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<vector<vector<vector<float>>>> *vec)
{
    int s1, s2, s3;
    int i, j, k;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        s2 = vec->at(i).size();
        for (j = 0; j < s2; ++j)
        {
            s3 = vec->at(i).at(j).size();

            for (k = 0; k < s3; ++k)
            {
                // vec->at(i).at(j).at(k).clear();
                vec->at(i).at(j).at(k).resize(0);
                vec->at(i).at(j).at(k).shrink_to_fit();
                // vec->at(i).at(j).at(k).~vector();
                // vector<float>().swap(vec->at(i).at(j).at(k));
            }
            // vec->at(i).at(j).clear();
            vec->at(i).at(j).resize(0);
            vec->at(i).at(j).shrink_to_fit();
        }
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<vector<vector<bool>>> *vec)
{
    int s1, s2;
    int i, j;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        s2 = vec->at(i).size();
        for (j = 0; j < s2; ++j)
        {
            // vec->at(i).at(j).clear();
            vec->at(i).at(j).resize(0);
            vec->at(i).at(j).shrink_to_fit();
            // vec->at(i).at(j).~vector();
            // vector<float>().swap(vec->at(i).at(j));
        }
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<vector<vector<float>>> *vec)
{
    int s1, s2;
    int i, j;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        s2 = vec->at(i).size();
        for (j = 0; j < s2; ++j)
        {
            // vec->at(i).at(j).clear();
            vec->at(i).at(j).resize(0);
            vec->at(i).at(j).shrink_to_fit();
            // vec->at(i).at(j).~vector();
            // vector<float>().swap(vec->at(i).at(j));
        }
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<vector<bool>> *vec)
{
    int s1;
    int i;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<vector<float>> *vec)
{
    int s1;
    int i;
    s1 = vec->size();
    for (i = 0; i < s1; ++i)
    {
        // vec->at(i).clear();
        vec->at(i).resize(0);
        vec->at(i).shrink_to_fit();
    }
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
}
void clearAndResizeVector(vector<bool> *vec)
{
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();

    // vec->~vector();
    // vector<bool>().swap(*vec);
}
void clearAndResizeVector(vector<float> *vec)
{
    // vec->clear();
    vec->resize(0);
    vec->shrink_to_fit();
    // vec->~vector();
    // vector<float>().swap(*vec);
}

/// @brief Generates a random uniform float value
/// @param range difference between the maximum and the minimum value
/// @param mean The mean of the distribution
/// @return Random float value with mean mean, and lowest possible value mean -0.5range and max mean +0.5range
float randomUniformFloatGenerator(float range, float mean)
{
    int top = 1000000;
    int bottom = top / range;
    float shift = ((float)top / bottom) / 2 - mean;
    return (((float)(rand() % top) / bottom) - shift);
}
float randomUniformFloatGenerator(double range, double mean)
{
    int top = 1000000;
    int bottom = top / range;
    float shift = ((float)top / bottom) / 2 - mean;
    return (((float)(rand() % top) / bottom) - shift);
}

/// @brief Caculates the sigmoid value of x with sigma
/// @param x 
/// @param sigma 
/// @return Sigmoid of x
float sigmoid(float x, float sigma){
    return 1.0/(1.0+exp(-sigma*x)); 
}