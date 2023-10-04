#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <vector>
#include <map>
#include <mutex>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include "omp.h"

#include <sys/stat.h>
#include <sys/time.h>

#include "./utilities.h"
#include "./CVRPGrapher.h"
using namespace std;

// TODO Create    data structure representing the genome possibility range

class GenomeCVRP
{
    // This will be the genome that will be used in the custom NEAT algorithm
    // We will assume a max of 3 hidden layers with a maximum of 100 nodes per layer as a starting point
    // The input for the genome will be a n*n (maxInputSize) of the current edge usage in a cplex node and the distance between customers (I0)
    // Note that both of these matrices are symmetrical and thus are joined into one. (This matrix remains expandable and thus future larger problems may be tackled)
    // The input also contains the matrix with coordinates and demands (3*n) (I1)
    // The input also contains a number of other information in a 1-d vector(1*s) (e.g. node depth, relaxation value, incumbent solution found) (I2)
    // The input contains one bias Integer (1) the weights should handle the size and direction of the bias (I3)

private:
    int maxNodesHiddenLayer = 500;
    int maxInputSize = 50;
    int s = 25;
    int bias = 1;
    float startingValue = 0.01;
    bool TEMPBOOL = 0;
    // These are the nodes of the first hidden layer, they will only hold the information during computation.
    vector<float> HLV1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    vector<float> HLV2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    vector<float> HLV3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    // This Vector will hold the ouput values
    vector<vector<float>> OUTPUT2D = twoDimensionVectorCreator(maxInputSize, maxInputSize, (float)0);

    // IL->OL
    //  These are the activation nodes between the input and the output layer. These are the only ones that are initialized (Hence the non zero activation function)
    vector<vector<vector<vector<bool>>>> ALI0OL = fourDimensionVectorCreator(maxInputSize, maxInputSize, maxInputSize, maxInputSize, (bool)1);
    vector<vector<vector<vector<bool>>>> ALI1OL = fourDimensionVectorCreator(3, maxInputSize, maxInputSize, maxInputSize, (bool)1);
    vector<vector<vector<vector<bool>>>> ALI2OL = fourDimensionVectorCreator(1, s, maxInputSize, maxInputSize, (bool)1);

    // These are the weight nodes between the input and the output layer. These are the only ones that are initialized (Hence the non zero activation function)
    vector<vector<vector<vector<float>>>> WLI0OL = fourDimensionVectorCreator(maxInputSize, maxInputSize, maxInputSize, maxInputSize, (float)1);
    vector<vector<vector<vector<float>>>> WLI1OL = fourDimensionVectorCreator(3, maxInputSize, maxInputSize, maxInputSize, (float)1);
    vector<vector<vector<vector<float>>>> WLI2OL = fourDimensionVectorCreator(1, s, maxInputSize, maxInputSize, (float)1);

    // IL->HL1
    //  These are the activation layers between the input and the first hidden layer
    vector<vector<vector<bool>>> ALI0H1 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H1 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H1 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<bool> ALI3H1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H1 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H1 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H1 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);
    vector<float> WLI3H1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

    // IL->HL2
    //  These are the activation layers between the input and the second hidden layer
    vector<vector<vector<bool>>> ALI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<bool> ALI3H2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);
    vector<float> WLI3H2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

    // IL->HL3
    //  These are the activation layers between the input and the third hidden layer
    vector<vector<vector<bool>>> ALI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<bool> ALI3H3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);
    vector<float> WLI3H3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

    // HL->HL
    //  These are the activation layers between the hidden layers 1-2, 1-3 and 2-3 respectively
    vector<vector<bool>> ALH1H2 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<bool>> ALH1H3 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<bool>> ALH2H3 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the hidden layers 1-2, 1-3 and 2-3 respectively
    vector<vector<float>> WLH1H2 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<float>> WLH1H3 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<float>> WLH2H3 = twoDimensionVectorCreator(maxNodesHiddenLayer, maxNodesHiddenLayer, (float)startingValue);

    // HL->OL
    //  These are the activation layers between the hidden layers and the output layer
    vector<vector<vector<bool>>> ALH1OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALH2OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALH3OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (bool)TEMPBOOL);
    // These are the weight layers between the hidden layers and the output layer
    vector<vector<vector<float>>> WLH1OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (float)startingValue);
    vector<vector<vector<float>>> WLH2OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (float)startingValue);
    vector<vector<vector<float>>> WLH3OL = threeDimensionVectorCreator(maxNodesHiddenLayer, maxInputSize, maxInputSize, (float)startingValue);

public:
    GenomeCVRP()
    {
        cout << "\n";
    }
    vector<int> feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<int>> *coordinates, vector<vector<float>> *misc);
};

vector<int> GenomeCVRP::feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<int>> *coordinates, vector<vector<float>> *misc)
{

    // Reset hidden layer values
    // step 1, calculate the values of the first hidden layer
    int i, j, k, l;
    cout << "\nFeeding Forward\n";
    // TIMER
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ios_base::sync_with_stdio(false);
    cout << "\n------------------------------******************HL1******************------------------------------\n";
    // Step 1.1 M1->H1

#pragma omp parallel for private(i, j, k) shared(HLV1, ALI0H1, WLI0H1, M1)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < maxInputSize; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV1[k] += M1->at(i).at(j) * ALI0H1[i][j][k] * WLI0H1[i][j][k];
            }
        }
    }

    // Step 1.2 coordinates -> H1

#pragma omp parallel for private(i, j, k) shared(HLV1, ALI1H1, WLI1H1, coordinates)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV1[k] += coordinates->at(i).at(j) * ALI1H1[i][j][k] * WLI1H1[i][j][k];
            }
        }
    }

    // Step 1.3 misc-> H1

#pragma omp parallel for private(i, k) shared(HLV1, ALI2H1, WLI2H1, misc)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < s; ++i)
        {
            HLV1[k] += misc->at(0).at(i) * ALI2H1[0][i][k] * WLI2H1[0][i][k];
        }
    }
    cout << "\n------------------------------******************HL2******************------------------------------\n";
    // step 2, calculate the values of the second hidden layer
    // Step 2.1 M1->H2

#pragma omp parallel for private(i, j, k) shared(HLV2, ALI0H2, WLI0H2, M1)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < maxInputSize; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV2[k] += M1->at(i).at(j) * ALI0H2[i][j][k] * WLI0H2[i][j][k];
            }
        }
    }

    // Step 2.2 coordinates -> H2

#pragma omp parallel for private(i, j, k) shared(HLV2, ALI1H2, WLI1H2, coordinates)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV2[k] += coordinates->at(i).at(j) * ALI1H2[i][j][k] * WLI1H2[i][j][k];
            }
        }
    }

    // Step 2.3 misc-> H2

#pragma omp parallel for private(i, k) shared(HLV2, ALI2H2, WLI2H2, misc)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < s; ++i)
        {
            HLV2[k] += misc->at(0).at(i) * ALI2H2[0][i][k] * WLI2H2[0][i][k];
        }
    }
    // Step 2.4 H1->H2

#pragma omp parallel for private(i, j) shared(HLV2, HLV1, ALH1H2, WLH1H2)

    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            HLV2[i] += HLV1[j] * ALH1H2[i][j] * WLH1H2[i][j];
        }
    }


    cout << "\n------------------------------******************HL3******************------------------------------\n";
    // Step 3, calculate the values of the third hidden layer
    // Step 3.1 M1->H3

#pragma omp parallel for private(i, j, k) shared(HLV3, ALI0H3, WLI0H3, M1)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < maxInputSize; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV3[k] += M1->at(i).at(j) * ALI0H3[i][j][k] * WLI0H3[i][j][k];
            }
        }
    }

    // Step 3.2 coordinates -> H3

#pragma omp parallel for private(i, j, k) shared(HLV3, ALI1H3, WLI1H3, coordinates)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < 3; ++i)
        {
            for (j = 0; j < maxInputSize; ++j)
            {
                HLV3[k] += coordinates->at(i).at(j) * ALI1H3[i][j][k] * WLI1H3[i][j][k];
            }
        }
    }

    // Step 3.3 misc-> H3

#pragma omp parallel for private(i, k) shared(HLV3, ALI2H3, WLI2H3, misc)
    for (k = 0; k < maxNodesHiddenLayer; ++k)
    {
        for (i = 0; i < s; ++i)
        {
            HLV3[k] += misc->at(0).at(i) * ALI2H3[0][i][k] * WLI2H3[0][i][k];
        }
    }

    // Step 3.4 H1->H3

#pragma omp parallel for private(i, j) shared(HLV3, HLV1, ALH1H3, WLH1H3)

    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            HLV3[i] += HLV1[j] * ALH1H3[i][j] * WLH1H3[i][j];
        }
    }

    // Step 3.5 H1->H3

#pragma omp parallel for private(i, j) shared(HLV3, HLV2, ALH1H3, WLH1H3)

    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            HLV3[i] += HLV2[j] * ALH1H3[i][j] * WLH1H3[i][j];
        }
    }


    cout << "\n------------------------------**************OUTPUTLAYER**************------------------------------\n"
         << flush;
    // Step 4, calculate the values of the output layer
    // Step OL.1 M1->OL
#pragma omp parallel for private(i, j, k, l) shared(OUTPUT2D, ALI0OL, WLI0OL, M1)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)
        {
            for (i = 0; i < maxInputSize; ++i)
            {
                for (j = 0; j < maxInputSize; ++j)
                {
                    OUTPUT2D[k][l] += M1->at(i).at(j) * ALI0OL[i][j][k][l] * WLI0OL[i][j][k][l];
                }
            }
        }
    }

    // Step OL.2 coordinates -> OL

#pragma omp parallel for private(i, j, k, l) shared(OUTPUT2D, ALI1OL, WLI1OL, coordinates)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)
        {
            for (i = 0; i < 3; ++i)
            {
                for (j = 0; j < maxInputSize; ++j)
                {
                    OUTPUT2D[k][l] += coordinates->at(i).at(j) * ALI1OL[i][j][k][l] * WLI1OL[i][j][k][l];
                }
            }
        }
    }

    // Step OL.3 misc-> OL

#pragma omp parallel for private(i, k, l) shared(OUTPUT2D, ALI2OL, WLI2OL, misc)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)
        {
            for (i = 0; i < s; ++i)
            {
                OUTPUT2D[k][l] += misc->at(0).at(i) * ALI2OL[0][i][k][l] * WLI2OL[0][i][k][l];
            }
        }
    }

    // Step OL.4 H1->OL

#pragma omp parallel for private(i, k, l) shared(OUTPUT2D, HLV1, ALH1OL, WLH1OL)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)

        {
            for (i = 0; i < maxNodesHiddenLayer; ++i)
            {
                OUTPUT2D[k][l] += HLV1[i] * ALH1OL[i][k][l] * WLH1OL[i][k][l];
            }
        }
    }

    // Step OL.5 H2->OL

#pragma omp parallel for private(i, k, l) shared(OUTPUT2D, HLV2, ALH2OL, WLH2OL)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)
        {
            for (i = 0; i < maxNodesHiddenLayer; ++i)
            {
                OUTPUT2D[k][l] += HLV2[i] * ALH2OL[i][k][l] * WLH2OL[i][k][l];
            }
        }
    }


    //     // Step OL.6 H3->OL

#pragma omp parallel for private(i, k, l) shared(OUTPUT2D, HLV3, ALH3OL, WLH3OL)
    for (k = 0; k < maxInputSize; ++k)
    {
        for (l = 0; l < maxInputSize; ++l)
        {
            for (i = 0; i < maxNodesHiddenLayer; ++i)
            {
                OUTPUT2D[k][l] += HLV3[i] * ALH3OL[i][k][l] * WLH3OL[i][k][l];
            }
        }
    }

    vector<int> OUTPUT1D(2, 0);
    float maxValue = 0;

    for (int i = 0; i < 50; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {

            OUTPUT2D[i][j] = (float)rand() * 1 / (float)RAND_MAX;
        }
    }

    // Step 5, get the highest value of the last layer and return the i an j coordinates of it
    // #pragma omp parallel for private(i, j) shared(OUTPUT2D, OUTPUT1D)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            if (OUTPUT2D[i][j] > maxValue)
            {
                maxValue = OUTPUT2D[i][j];
                OUTPUT1D[0] = i;
                OUTPUT1D[1] = j;
            }
        }
    }
#pragma omp parallel for private(i, j) shared(OUTPUT2D, OUTPUT1D)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            if (OUTPUT2D[i][j] > maxValue)
            {

                maxValue = OUTPUT2D[i][j];
                OUTPUT1D[0] = i;
                OUTPUT1D[1] = j;
            }
        }
    }
    gettimeofday(&end, NULL);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec -
                                start.tv_usec)) *
                 1e-6;
    cout << "\nTime taken by program is : " << fixed
         << time_taken << setprecision(6);
    cout << " sec" << endl;

    return OUTPUT1D;
}
// TODO Create function to calculate the distance between two individuals (species representative and current individual)

// TODO Create function to determine an individuals fitness

// TODO Create fucntion that will run the cplex with the individuals AI as branching scheme

// TODO Create function that will run the cplex with the standar CPLEX branching scheme

// TODO Create function that reports and log what happened in the current generation (one log file is better to use later)

// TODO Create function that will generate an ofspring from two individuals

// TODO Create function that will eliminate x% of the worst individuals of the population

int main()
{
    srand((unsigned int)time(NULL));
    GenomeCVRP g0;
    vector<vector<float>> M1 = twoDimensionVectorCreator(50, 50, (float)1);
    vector<vector<int>> coordinates = twoDimensionVectorCreator(3, 50, 1);
    vector<vector<float>> misc = twoDimensionVectorCreator(1, 25, (float)1);
    int bias = 1;

    for (int i = 0; i < 50; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {

            M1[i][j] = (float)rand() * 1 / (float)RAND_MAX;
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {
            coordinates[i][j] = (int)rand() % 2;
        }
    }

    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < 25; ++j)
        {
            misc[i][j] = (float)rand() * 2 / (float)RAND_MAX;
        }
    }

    g0.feedForwardFirstTry(&M1, &coordinates, &misc);

    return 0;
}