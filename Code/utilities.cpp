#include "./utilities.h"
/// This file contains all the helper functions, including custom printer functions and the graph creator functions

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
/// @param filePath Where to save the file
/// @param filename Filename (not used to save the file, only written in it)
void writeCVRP(vector<vector<int>> nodeCoordinatesAndDemand, int capacity, string filePath, string filename)
{
    ofstream fw(filePath, ofstream::out);
    if (!fw.is_open())
    {
        return;
    }
    int rowNumber = nodeCoordinatesAndDemand.size();
    fw << "NAME:        " << filename << "\n";
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
        int s = (rand() % 6) + 3; // This generates a random number between 3 and 8
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
    { // Cluster and random method
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
            int j = 0;

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

