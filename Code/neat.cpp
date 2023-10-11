#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <vector>
#include <list>
#include <map>
#include <mutex>
#include <ctime>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include "omp.h"
#include <sys/stat.h>
#include <sys/time.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/cplex.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/ilocplex.h>

#include "./utilities.h"
#include "./CVRPGrapher.h"
// #include "./branchAndCut.h"
using namespace std;

std::mutex mtxCVRP; // mutex for critical section

#define PERTURBEPROB 0.1       // Probability that the weight of a gene is perturbed (in reality this is PERTURBEPROB-MUTATEPROB)
#define MUTATEPROB 0.01        // or that a complete new value is given
#define NEWNODEPROB 0.03       // or that a complete new value is given
#define NEWCONNECTIONPROB 0.05 // or that a complete new value is given

typedef IloArray<IloNumVarArray> NumVarMatrix;
typedef IloArray<IloNumArray> NumMatrix;

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
    int maxInputSize = 15;
    int s = 25;
    int bias = 1;
    float startingValue = 0.01;
    bool TEMPBOOL = 0;
    // These are the nodes of the hidden layers, they will only hold the information during computation.
    mutable vector<float> HLV1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    mutable vector<float> HLV2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    mutable vector<float> HLV3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);

    // These are the nodes of the first hidden layer, they are boolean true if the node exists.
    mutable vector<bool> HLA1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)0);
    mutable vector<bool> HLA2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)0);
    mutable vector<bool> HLA3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (bool)0);
    // This Vector will hold the ouput values

    mutable vector<vector<float>> OUTPUT2D = twoDimensionVectorCreator(maxInputSize, maxInputSize, (float)0);

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
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H1 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H1 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H1 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);

    // IL->HL2
    //  These are the activation layers between the input and the second hidden layer
    vector<vector<vector<bool>>> ALI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);

    // IL->HL3
    //  These are the activation layers between the input and the third hidden layer
    vector<vector<vector<bool>>> ALI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);

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
    int fitness = 0;
    int specieNumber = 0;
    GenomeCVRP();
    vector<int> feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<float>> *coordinates, vector<vector<float>> *misc) const;
    void mutateGenomeWeights();
    void mutateGenomeNodeStructure();
    void mutateGenomeConnectionStructure();
    void setFitness(int value);
    int getFitness();
};
class PopulationCVRP
{

private:
    int populationSize = 10;
    GenomeCVRP population[1];
    GenomeCVRP speciesRepresentative[4]; // Each species needs a representative from the previous generation, such that new genomes can check if they belong

public:
    int generation = 12;
    vector<vector<int>> vertexMatrix;
    int capacity;
    vector<vector<int>> costMatrix;

    PopulationCVRP();
    void getCVRP(int nCostumers, string depotLocation, string customerDistribution, int demandDistribution);
    void getCVRP(string filepath);
    void runCPLEXGenome();
};
class CVRPCallback : public IloCplex::Callback::Function
{
private:
    // Variables for edge Usage.
    NumVarMatrix edgeUsage;

    // Capacity Variable
    IloInt Q;

    // Demand vector (depot is always first and has demand 0)
    vector<int> demandVector;

    // Matrix size (Customers + depot)
    IloInt N;

    // Filepath where the data is saved to.
    string filepathCGC;

    // If we are in training we will save each node's instance
    mutable bool training;

    // tracks how often we have made a cut here
    IloInt cutCalls = 0;
    mutable IloInt branches = 0;
    GenomeCVRP genome;

    // These are the input to the AI
    mutable vector<vector<float>> M1;
    mutable vector<vector<float>> coordinates;
    mutable vector<vector<float>> misc;

public:
    // Constructor with data.
    CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                 const NumVarMatrix &_edgeUsage, vector<vector<int>> &_costMatrix, vector<vector<int>> &_coordinates, GenomeCVRP &_genome);

    inline void
    connectedComponents(const IloCplex::Callback::Context &context) const;
    inline void
    branchingWithGenome(const IloCplex::Callback::Context &context) const;

    int getCalls() const;
    int getBranches() const;
    virtual void invoke(const IloCplex::Callback::Context &context) ILO_OVERRIDE;
    virtual ~CVRPCallback();
};

tuple<vector<vector<int>>, int> createAndRunCPLEXInstance(vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity, bool training, GenomeCVRP genome);

//----------**********GENOME*****-----
GenomeCVRP::GenomeCVRP()
{
    cout << "";
}

vector<int> GenomeCVRP::feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<float>> *coordinates, vector<vector<float>> *misc) const
{

    // Reset hidden layer values
    // step 1, calculate the values of the first hidden layer
    int i, j, k, l;
    // TIMER
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ios_base::sync_with_stdio(false);
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

    // Step 6 convert 2d to a vector of i and j
    vector<int> OUTPUT1D(2, 0);
    float maxValue = 0;
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
    // cout << "\nTime taken by program is : " << fixed
    //      << time_taken << setprecision(6);
    // cout << " sec" << endl << flush;

    return OUTPUT1D;
}

// Mutates the genome
void GenomeCVRP::mutateGenomeWeights()
{
    // Mutate weights (%of weights are perturbed and % are given random new value)
    // We do this for every active connection
    float randFloat;
    int i, j, k, l;

    // ALI0OL
#pragma omp parallel for private(i, j, k, l) shared(ALI0OL, WLI0OL)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI0OL[i][j][k][l] == 1)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat <= MUTATEPROB)
                        {
                            // If we mutate we get a complete new random value a random float value between -1 and 1
                            WLI0OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                            continue;
                        }
                        if (randFloat < PERTURBEPROB)
                        {
                            // If we perturbe by a value between -0.1 & 0.1
                            WLI0OL[i][j][k][l] += ((float)(rand() % 100000) / 1000000) - 1;
                        }
                    }
                }
            }
        }
    }

    // ALI1OL
#pragma omp parallel for private(i, j, k, l) shared(ALI1OL, WLI1OL)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI1OL[i][j][k][l] == 1)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat <= MUTATEPROB)
                        {
                            // If we mutate we get a complete new random value a random float value between -1 and 1
                            WLI1OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                            continue;
                        }
                        if (randFloat < PERTURBEPROB)
                        {
                            // If we perturbe by a value between -0.1 & 0.1
                            WLI1OL[i][j][k][l] += ((float)(rand() % 100000) / 1000000) - 1;
                        }
                    }
                }
            }
        }
    }

    // ALI2OL
#pragma omp parallel for private(i, j, k, l) shared(ALI2OL, WLI2OL)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI2OL[i][j][k][l] == 1)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat <= MUTATEPROB)
                        {
                            // If we mutate we get a complete new random value a random float value between -1 and 1
                            WLI2OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                            continue;
                        }
                        if (randFloat < PERTURBEPROB)
                        {
                            // If we perturbe by a value between -0.1 & 0.1
                            WLI2OL[i][j][k][l] += ((float)(rand() % 100000) / 1000000) - 1;
                        }
                    }
                }
            }
        }
    }

    // ALI0H1
#pragma omp parallel for private(i, j, k) shared(ALI0H1, WLI0H1)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI0H1[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI0H1[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI0H1[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALI1H1
#pragma omp parallel for private(i, j, k) shared(ALI1H1, WLI1H1)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI1H1[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI1H1[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI1H1[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALI2H1
#pragma omp parallel for private(i, j, k) shared(ALI2H1, WLI2H1)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI2H1[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI2H1[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI2H1[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALI0H2
#pragma omp parallel for private(i, j, k) shared(ALI0H2, WLI0H2)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI0H2[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI0H2[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI0H2[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    // ALI1H2
#pragma omp parallel for private(i, j, k) shared(ALI1H2, WLI1H2)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI1H2[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI1H2[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI1H2[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALI2H2
#pragma omp parallel for private(i, j, k) shared(ALI2H2, WLI2H2)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI2H2[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI2H2[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI2H2[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALI0H3
#pragma omp parallel for private(i, j, k) shared(ALI0H3, WLI0H3)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI0H3[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI0H3[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI0H3[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    // ALI1H3
#pragma omp parallel for private(i, j, k) shared(ALI1H3, WLI1H3)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI1H3[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI1H3[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI1H3[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }
    //  ALI2H3
#pragma omp parallel for private(i, j, k) shared(ALI2H3, WLI2H3)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI2H3[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLI2H3[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLI2H3[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    //  ALH1H2
#pragma omp parallel for private(i, j) shared(ALH1H2, WLH1H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH1H2[i][j] == 1)
            {
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat <= MUTATEPROB)
                {
                    // If we mutate we get a complete new random value a random float value between -1 and 1
                    WLH1H2[i][j] = ((float)(rand() % 100000) / 50000) - 1;
                    continue;
                }
                if (randFloat < PERTURBEPROB)
                {
                    // If we perturbe by a value between -0.1 & 0.1
                    WLH1H2[i][j] += ((float)(rand() % 100000) / 1000000) - 1;
                }
            }
        }
    }

    // ALH1H3
#pragma omp parallel for private(i, j) shared(ALH1H3, WLH1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH1H3[i][j] == 1)
            {
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat <= MUTATEPROB)
                {
                    // If we mutate we get a complete new random value a random float value between -1 and 1
                    WLH1H3[i][j] = ((float)(rand() % 100000) / 50000) - 1;
                    continue;
                }
                if (randFloat < PERTURBEPROB)
                {
                    // If we perturbe by a value between -0.1 & 0.1
                    WLH1H3[i][j] += ((float)(rand() % 100000) / 1000000) - 1;
                }
            }
        }
    }

    //  ALH2H3
#pragma omp parallel for private(i, j) shared(ALH2H3, WLH2H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH2H3[i][j] == 1)
            {
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat <= MUTATEPROB)
                {
                    // If we mutate we get a complete new random value a random float value between -1 and 1
                    WLH2H3[i][j] = ((float)(rand() % 100000) / 50000) - 1;
                    continue;
                }
                if (randFloat < PERTURBEPROB)
                {
                    // If we perturbe by a value between -0.1 & 0.1
                    WLH2H3[i][j] += ((float)(rand() % 100000) / 1000000) - 1;
                }
            }
        }
    }

    //  ALH1OL
#pragma omp parallel for private(i, j, k) shared(ALH1OL, WLH1OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH1OL[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLH1OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLH1OL[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    // ALH2OL
#pragma omp parallel for private(i, j, k) shared(ALH2OL, WLH2OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH2OL[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLH2OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLH2OL[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }

    // ALH3OL
#pragma omp parallel for private(i, j, k) shared(ALH3OL, WLH3OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH3OL[i][j][k] == 1)
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat <= MUTATEPROB)
                    {
                        // If we mutate we get a complete new random value a random float value between -1 and 1
                        WLH3OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                        continue;
                    }
                    if (randFloat < PERTURBEPROB)
                    {
                        // If we perturbe by a value between -0.1 & 0.1
                        WLH3OL[i][j][k] += ((float)(rand() % 100000) / 1000000) - 1;
                    }
                }
            }
        }
    }
}

void GenomeCVRP::mutateGenomeNodeStructure()
{
    // Mutate connection structure.
    // For a new node to be created an existing connection must exist between two nodes, and the nodes must not be in neighbouring layers
    //(e.g. a connection between the input and the first hidden layer cannot be split)
    float randFloat;
    int i, j, k, l, newNode;

    // IL0->HL2
#pragma omp parallel for private(i, j, k, l) shared(ALI0H2, WLI0H2, ALI0H1, WLI0H1, ALH1H2, WLH1H2, HLA1)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI0H2[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {
                    if (HLA1[l] == false)
                    { // node k is not yet active
                        newNode = k;
                        HLA1[l] = true;
                        freeNode = true;
                        break;
                    }
                }
                if (!freeNode) // No nodes can be added anymore
                {
                    continue;
                }
                ALI0H2[i][j][k] = false; // This connection no longer exists, as it is split

                ALI0H1[i][j][l] = true;            // The connection is now going to node l in the first hidden layer
                WLI0H1[i][j][l] = WLI0H2[i][j][k]; // The first weight is equal to the previous connection weight

                ALH1H2[l][k] = true;
                WLH1H2[l][k] = 1; // and for the second it is 1, effectively not changing the values initially
            }
        }
    }

    // IL1->HL2
#pragma omp parallel for private(i, j, k, l) shared(ALI1H2, WLI1H2, ALI1H1, WLI1H1, ALH1H2, WLH1H2, HLA1)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI1H2[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {
                    if (HLA1[l] == false)
                    { // node k is not yet active
                        newNode = k;
                        HLA1[l] = true;
                        freeNode = true;
                        break;
                    }
                }
                if (!freeNode) // No nodes can be added anymore
                {
                    continue;
                }
                ALI1H2[i][j][k] = false; // This connection no longer exists, as it is split

                ALI1H1[i][j][l] = true;            // The connection is now going to node l in the first hidden layer
                WLI1H1[i][j][l] = WLI1H2[i][j][k]; // The first weight is equal to the previous connection weight

                ALH1H2[l][k] = true;
                WLH1H2[l][k] = 1; // and for the second it is 1, effectively not changing the values initially
            }
        }
    }

    // IL2->HL2
#pragma omp parallel for private(i, j, k, l) shared(ALI2H2, WLI2H2, ALI2H1, WLI2H1, ALH1H2, WLH1H2, HLA1)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI2H2[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {
                    if (HLA1[l] == false)
                    { // node k is not yet active
                        newNode = k;
                        HLA1[l] = true;
                        freeNode = true;
                        break;
                    }
                }
                if (!freeNode) // No nodes can be added anymore
                {
                    continue;
                }
                ALI2H2[i][j][k] = false; // This connection no longer exists, as it is split

                ALI2H1[i][j][l] = true;            // The connection is now going to node l in the first hidden layer
                WLI2H1[i][j][l] = WLI2H2[i][j][k]; // The first weight is equal to the previous connection weight

                ALH1H2[l][k] = true;
                WLH1H2[l][k] = 1; // and for the second it is 1, effectively not changing the values initially
            }
        }
    }

    int l1, l2, l3;
// IL0->HL3
#pragma omp parallel for private(i, j, k, l) shared(ALI0H3, WLI0H3, ALI0H1, WLI0H1, ALI0H2, WLI0H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI0H3[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode1 = false;
                bool freeNode2 = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {

                    if (HLA1[l] == false)
                    { // node k is not yet active
                        l1 = k;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node k is not yet active
                        l2 = k;
                        freeNode2 = true;
                    }
                }
                if (freeNode1 && freeNode2) // Node can be added in both layers
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > 0.5)
                    {
                        HLA1[l1] = true;
                        ALI0H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI0H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                        WLI0H1[i][j][l1] = WLI0H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH1H3[l2][k] = true;
                        WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                    else
                    {
                        HLA2[l2] = true;
                        ALI0H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI0H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLI0H2[i][j][l2] = WLI0H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH2H3[l2][k] = true;
                        WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                }
                else if (freeNode1) // New node in the first hidden layer
                {
                    HLA1[l1] = true;
                    ALI0H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI0H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                    WLI0H1[i][j][l1] = WLI0H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH1H3[l2][k] = true;
                    WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                else if (freeNode2) // New node in the second hidden layer
                {
                    HLA2[l2] = true;
                    ALI0H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI0H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLI0H2[i][j][l2] = WLI0H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH2H3[l2][k] = true;
                    WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                // Else do nothing as there are no places for new nodes
            }
        }
    }

// IL1->HL3
#pragma omp parallel for private(i, j, k, l) shared(ALI1H3, WLI1H3, ALI1H1, WLI1H1, ALI1H2, WLI1H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI1H3[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode1 = false;
                bool freeNode2 = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {

                    if (HLA1[l] == false)
                    { // node k is not yet active
                        l1 = k;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node k is not yet active
                        l2 = k;
                        freeNode2 = true;
                    }
                }
                if (freeNode1 && freeNode2) // Node can be added in both layers
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > 0.5)
                    {
                        HLA1[l1] = true;
                        ALI1H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI1H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                        WLI1H1[i][j][l1] = WLI1H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH1H3[l2][k] = true;
                        WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                    else
                    {
                        HLA2[l2] = true;
                        ALI1H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI1H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLI1H2[i][j][l2] = WLI1H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH2H3[l2][k] = true;
                        WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                }
                else if (freeNode1) // New node in the first hidden layer
                {
                    HLA1[l1] = true;
                    ALI1H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI1H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                    WLI1H1[i][j][l1] = WLI1H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH1H3[l2][k] = true;
                    WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                else if (freeNode2) // New node in the second hidden layer
                {
                    HLA2[l2] = true;
                    ALI1H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI1H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLI1H2[i][j][l2] = WLI1H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH2H3[l2][k] = true;
                    WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                // Else do nothing as there are no places for new nodes
            }
        }
    }

    // IL2->HL3
#pragma omp parallel for private(i, j, k, l) shared(ALI2H3, WLI2H3, ALI2H1, WLI2H1, ALI2H2, WLI2H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxNodesHiddenLayer; ++k)
            {
                if (ALI2H3[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode1 = false;
                bool freeNode2 = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {

                    if (HLA1[l] == false)
                    { // node k is not yet active
                        l1 = k;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node k is not yet active
                        l2 = k;
                        freeNode2 = true;
                    }
                }
                if (freeNode1 && freeNode2) // Node can be added in both layers
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > 0.5)
                    {
                        HLA1[l1] = true;
                        ALI2H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI2H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                        WLI2H1[i][j][l1] = WLI2H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH1H3[l2][k] = true;
                        WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                    else
                    {
                        HLA2[l2] = true;
                        ALI2H3[i][j][k] = false; // This connection no longer exists, as it is split

                        ALI2H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLI2H2[i][j][l2] = WLI2H3[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH2H3[l2][k] = true;
                        WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                }
                else if (freeNode1) // New node in the first hidden layer
                {
                    HLA1[l1] = true;
                    ALI2H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI2H1[i][j][l1] = true;            // The connection is now going to node l in the first hidden layer
                    WLI2H1[i][j][l1] = WLI2H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH1H3[l2][k] = true;
                    WLH1H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                else if (freeNode2) // New node in the second hidden layer
                {
                    HLA2[l2] = true;
                    ALI2H3[i][j][k] = false; // This connection no longer exists, as it is split

                    ALI2H2[i][j][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLI2H2[i][j][l2] = WLI2H3[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH2H3[l2][k] = true;
                    WLH2H3[l2][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                // Else do nothing as there are no places for new nodes
            }
        }
    }

    // IL0->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI0OL, WLI0OL, ALI0H1, WLI0H1, ALI0H2, WLI0H2, ALI0H3, WLI0H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH2OL, HLA1, HLA2, HLA3)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (int m = 0; m < maxInputSize; ++m)
                {
                    if (ALI0OL[i][j][k][m] == 0) // If there is no connection we cannot split it
                    {
                        continue;
                    }
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                    {
                        continue;
                    }
                    bool freeNode1 = false;
                    bool freeNode2 = false;
                    bool freeNode3 = false;
                    for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                    {

                        if (HLA1[l] == false)
                        { // node k is not yet active
                            l1 = k;
                            freeNode1 = true;
                        }
                        if (HLA2[l] == false)
                        { // node k is not yet active
                            l2 = k;
                            freeNode2 = true;
                        }
                        if (HLA3[l] == false)
                        { // node k is not yet active
                            l3 = k;
                            freeNode3 = true;
                        }
                    }
                    if (freeNode1 && freeNode2 && freeNode3) // Node can be added in all three layers
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.66)
                        {
                            HLA1[l1] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H1[i][j][l1] = true;
                            WLI0H1[i][j][l1] = WLI0OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else if (randFloat > 0.33)
                        {
                            HLA2[l2] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H2[i][j][l2] = true;
                            WLI0H2[i][j][l2] = WLI0OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H3[i][j][l3] = true;
                            WLI0H3[i][j][l3] = WLI0OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode2)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H1[i][j][l1] = true;
                            WLI0H1[i][j][l1] = WLI0OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA2[l2] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H2[i][j][l2] = true;
                            WLI0H2[i][j][l2] = WLI0OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H1[i][j][l1] = true;
                            WLI0H1[i][j][l1] = WLI0OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H3[i][j][l3] = true;
                            WLI0H3[i][j][l3] = WLI0OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode2 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA2[l2] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H2[i][j][l2] = true;
                            WLI0H2[i][j][l2] = WLI0OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI0OL[i][j][k][m] = false;

                            ALI0H3[i][j][l3] = true;
                            WLI0H3[i][j][l3] = WLI0OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1)
                    {
                        HLA1[l1] = true;
                        ALI0OL[i][j][k][m] = false;

                        ALI0H1[i][j][l1] = true;
                        WLI0H1[i][j][l1] = WLI0OL[i][j][k][m];

                        ALH1OL[l1][k][m] = true;
                        WLH1OL[l1][k][m] = 1;
                    }
                    else if (freeNode2)
                    {
                        HLA2[l2] = true;
                        ALI0OL[i][j][k][m] = false;

                        ALI0H2[i][j][l2] = true;
                        WLI0H2[i][j][l2] = WLI0OL[i][j][k][m];

                        ALH2OL[l2][k][m] = true;
                        WLH2OL[l2][k][m] = 1;
                    }
                    else if (freeNode3)
                    {
                        HLA3[l3] = true;
                        ALI0OL[i][j][k][m] = false;

                        ALI0H3[i][j][l3] = true;
                        WLI0H3[i][j][l3] = WLI0OL[i][j][k][m];

                        ALH3OL[l3][k][m] = true;
                        WLH3OL[l3][k][m] = 1;
                    }
                    // Else do nothing as there are no places for new nodes
                }
            }
        }
    }

    // IL1->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI1OL, WLI1OL, ALI1H1, WLI1H1, ALI1H2, WLI1H2, ALI1H3, WLI1H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH2OL, HLA1, HLA2, HLA3)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (int m = 0; m < maxInputSize; ++m)
                {
                    if (ALI1OL[i][j][k][m] == 0) // If there is no connection we cannot split it
                    {
                        continue;
                    }
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                    {
                        continue;
                    }
                    bool freeNode1 = false;
                    bool freeNode2 = false;
                    bool freeNode3 = false;
                    for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                    {

                        if (HLA1[l] == false)
                        { // node k is not yet active
                            l1 = k;
                            freeNode1 = true;
                        }
                        if (HLA2[l] == false)
                        { // node k is not yet active
                            l2 = k;
                            freeNode2 = true;
                        }
                        if (HLA3[l] == false)
                        { // node k is not yet active
                            l3 = k;
                            freeNode3 = true;
                        }
                    }
                    if (freeNode1 && freeNode2 && freeNode3) // Node can be added in all three layers
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.66)
                        {
                            HLA1[l1] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H1[i][j][l1] = true;
                            WLI1H1[i][j][l1] = WLI1OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else if (randFloat > 0.33)
                        {
                            HLA2[l2] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H2[i][j][l2] = true;
                            WLI1H2[i][j][l2] = WLI1OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H3[i][j][l3] = true;
                            WLI1H3[i][j][l3] = WLI1OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode2)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H1[i][j][l1] = true;
                            WLI1H1[i][j][l1] = WLI1OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA2[l2] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H2[i][j][l2] = true;
                            WLI1H2[i][j][l2] = WLI1OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H1[i][j][l1] = true;
                            WLI1H1[i][j][l1] = WLI1OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H3[i][j][l3] = true;
                            WLI1H3[i][j][l3] = WLI1OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode2 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA2[l2] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H2[i][j][l2] = true;
                            WLI1H2[i][j][l2] = WLI1OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI1OL[i][j][k][m] = false;

                            ALI1H3[i][j][l3] = true;
                            WLI1H3[i][j][l3] = WLI1OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1)
                    {
                        HLA1[l1] = true;
                        ALI1OL[i][j][k][m] = false;

                        ALI1H1[i][j][l1] = true;
                        WLI1H1[i][j][l1] = WLI1OL[i][j][k][m];

                        ALH1OL[l1][k][m] = true;
                        WLH1OL[l1][k][m] = 1;
                    }
                    else if (freeNode2)
                    {
                        HLA2[l2] = true;
                        ALI1OL[i][j][k][m] = false;

                        ALI1H2[i][j][l2] = true;
                        WLI1H2[i][j][l2] = WLI1OL[i][j][k][m];

                        ALH2OL[l2][k][m] = true;
                        WLH2OL[l2][k][m] = 1;
                    }
                    else if (freeNode3)
                    {
                        HLA3[l3] = true;
                        ALI1OL[i][j][k][m] = false;

                        ALI1H3[i][j][l3] = true;
                        WLI1H3[i][j][l3] = WLI1OL[i][j][k][m];

                        ALH3OL[l3][k][m] = true;
                        WLH3OL[l3][k][m] = 1;
                    }
                    // Else do nothing as there are no places for new nodes
                }
            }
        }
    }

    // IL2->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI2OL, WLI2OL, ALI2H1, WLI2H1, ALI2H2, WLI2H2, ALI2H3, WLI2H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH2OL, HLA1, HLA2, HLA3)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (int m = 0; m < maxInputSize; ++m)
                {
                    if (ALI2OL[i][j][k][m] == 0) // If there is no connection we cannot split it
                    {
                        continue;
                    }
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                    {
                        continue;
                    }
                    bool freeNode1 = false;
                    bool freeNode2 = false;
                    bool freeNode3 = false;
                    for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                    {

                        if (HLA1[l] == false)
                        { // node k is not yet active
                            l1 = k;
                            freeNode1 = true;
                        }
                        if (HLA2[l] == false)
                        { // node k is not yet active
                            l2 = k;
                            freeNode2 = true;
                        }
                        if (HLA3[l] == false)
                        { // node k is not yet active
                            l3 = k;
                            freeNode3 = true;
                        }
                    }
                    if (freeNode1 && freeNode2 && freeNode3) // Node can be added in all three layers
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.66)
                        {
                            HLA1[l1] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H1[i][j][l1] = true;
                            WLI2H1[i][j][l1] = WLI2OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else if (randFloat > 0.33)
                        {
                            HLA2[l2] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H2[i][j][l2] = true;
                            WLI2H2[i][j][l2] = WLI2OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H3[i][j][l3] = true;
                            WLI2H3[i][j][l3] = WLI2OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode2)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H1[i][j][l1] = true;
                            WLI2H1[i][j][l1] = WLI2OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA2[l2] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H2[i][j][l2] = true;
                            WLI2H2[i][j][l2] = WLI2OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                    }
                    else if (freeNode1 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA1[l1] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H1[i][j][l1] = true;
                            WLI2H1[i][j][l1] = WLI2OL[i][j][k][m];

                            ALH1OL[l1][k][m] = true;
                            WLH1OL[l1][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H3[i][j][l3] = true;
                            WLI2H3[i][j][l3] = WLI2OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode2 && freeNode3)
                    {
                        randFloat = (float)(rand() % 100000) / 100000;
                        if (randFloat > 0.5)
                        {
                            HLA2[l2] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H2[i][j][l2] = true;
                            WLI2H2[i][j][l2] = WLI2OL[i][j][k][m];

                            ALH2OL[l2][k][m] = true;
                            WLH2OL[l2][k][m] = 1;
                        }
                        else
                        {
                            HLA3[l3] = true;
                            ALI2OL[i][j][k][m] = false;

                            ALI2H3[i][j][l3] = true;
                            WLI2H3[i][j][l3] = WLI2OL[i][j][k][m];

                            ALH3OL[l3][k][m] = true;
                            WLH3OL[l3][k][m] = 1;
                        }
                    }
                    else if (freeNode1)
                    {
                        HLA1[l1] = true;
                        ALI2OL[i][j][k][m] = false;

                        ALI2H1[i][j][l1] = true;
                        WLI2H1[i][j][l1] = WLI2OL[i][j][k][m];

                        ALH1OL[l1][k][m] = true;
                        WLH1OL[l1][k][m] = 1;
                    }
                    else if (freeNode2)
                    {
                        HLA2[l2] = true;
                        ALI2OL[i][j][k][m] = false;

                        ALI2H2[i][j][l2] = true;
                        WLI2H2[i][j][l2] = WLI2OL[i][j][k][m];

                        ALH2OL[l2][k][m] = true;
                        WLH2OL[l2][k][m] = 1;
                    }
                    else if (freeNode3)
                    {
                        HLA3[l3] = true;
                        ALI2OL[i][j][k][m] = false;

                        ALI2H3[i][j][l3] = true;
                        WLI2H3[i][j][l3] = WLI2OL[i][j][k][m];

                        ALH3OL[l3][k][m] = true;
                        WLH3OL[l3][k][m] = 1;
                    }
                    // Else do nothing as there are no places for new nodes
                }
            }
        }
    }

    // HL1->HL3
#pragma omp parallel for private(i, j, l) shared(ALH1H3, WLH1H3, ALH1H2, WLH1H2, ALH2H3, WLH2H3, HLA2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH1H3[i][j] == 0) // If there is no connection we cannot split it
            {
                continue;
            }
            randFloat = (float)(rand() % 100000) / 100000;
            if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
            {
                continue;
            }
            bool freeNode = false;
            for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
            {
                if (HLA2[l] == false)
                { // node k is not yet active
                    newNode = k;
                    HLA2[l] = true;
                    freeNode = true;
                    break;
                }
            }
            if (!freeNode) // No nodes can be added anymore
            {
                continue;
            }
            ALH1H3[i][j] = false; // This connection no longer exists, as it is split

            ALH1H2[i][l] = true;         // The connection is now going to node l in the first hidden layer
            WLH1H2[i][l] = WLH1H3[i][j]; // The first weight is equal to the previous connection weight

            ALH2H3[l][j] = true;
            WLH2H3[l][j] = 1; // and for the second it is 1, effectively not changing the values initially
        }
    }

    // HL1->OL
#pragma omp parallel for private(i, j, k, l) shared(ALH1OL, WLH1OL, ALH1H2, WLH1H2, ALH2OL, WLH2OL, ALH1H3, WLH1H3, ALH3OL, WLH3OL, HLA2, HLA3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH1OL[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode2 = false;
                bool freeNode3 = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {
                    if (HLA2[l] == false)
                    { // node k is not yet active
                        l2 = k;
                        freeNode2 = true;
                    }
                    if (HLA3[l] == false)
                    { // node k is not yet active
                        l3 = k;
                        freeNode3 = true;
                    }
                }
                if (freeNode2 && freeNode3) // Node can be added in both layers
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > 0.5)
                    {
                        HLA2[l2] = true;
                        ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                        ALH1H2[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLH1H2[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH2OL[l2][j][k] = true;
                        WLH2OL[l2][j][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                    else
                    {
                        HLA3[l3] = true;
                        ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                        ALH1H3[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLH1H3[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH3OL[l2][j][k] = true;
                        WLH3OL[l2][j][k] = 1; // and for the second it is 1, effectively not changing the values initially
                    }
                }
                else if (freeNode2) // New node in the first hidden layer
                {
                    HLA2[l2] = true;
                    ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                    ALH1H2[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLH1H2[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH2OL[l2][j][k] = true;
                    WLH2OL[l2][j][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                else if (freeNode3) // New node in the second hidden layer
                {
                    HLA3[l3] = true;
                    ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                    ALH1H3[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLH1H3[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH3OL[l2][j][k] = true;
                    WLH3OL[l2][j][k] = 1; // and for the second it is 1, effectively not changing the values initially
                }
                // Else do nothing as there are no places for new nodes
            }
        }
    }
    // HL2->OL
#pragma omp parallel for private(i, j, k, l) shared(ALH2OL, WLH2OL, ALH2H3, WLH2H3, ALH3OL, WLH3OL, HLA3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH2OL[i][j][k] == 0) // If there is no connection we cannot split it
                {
                    continue;
                }
                randFloat = (float)(rand() % 100000) / 100000;
                if (randFloat > NEWNODEPROB) // The random number is higher than the treshold and we do not mutate
                {
                    continue;
                }
                bool freeNode = false;
                for (l = 0; l < maxNodesHiddenLayer; ++l) // Find a node in the hidden layer that is not yet used
                {
                    if (HLA3[l] == false)
                    { // node k is not yet active
                        newNode = k;
                        HLA3[l] = true;
                        freeNode = true;
                        break;
                    }
                }
                if (!freeNode) // No nodes can be added anymore
                {
                    continue;
                }
                ALH2OL[i][j][k] = false; // This connection no longer exists, as it is split

                ALH2H3[i][l] = true;            // The connection is now going to node l in the first hidden layer
                WLH2H3[i][l] = WLH2OL[i][j][k]; // The first weight is equal to the previous connection weight

                ALH3OL[l][j][k] = true;
                WLH3OL[l][j][k] = 1; // and for the second it is 1, effectively not changing the values initially
                                     // Else do nothing as there are no places for new nodes
            }
        }
    }
}

void GenomeCVRP::mutateGenomeConnectionStructure()
{
    int i, j, k, l;
    // IL0->H1
#pragma omp parallel for private(i, j, k) shared(HLA1, ALI0H1, WLI0H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H1[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI0H1[j][k][i] = true;
                    WLI0H1[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL1->H1
#pragma omp parallel for private(i, j, k) shared(HLA1, ALI1H1, WLI1H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H1[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI1H1[j][k][i] = true;
                    WLI1H1[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL2->H1
#pragma omp parallel for private(i, j, k) shared(HLA1, ALI2H1, WLI2H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H1[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI2H1[j][k][i] = true;
                    WLI2H1[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL0->H2
#pragma omp parallel for private(i, j, k) shared(HLA2, ALI0H2, WLI0H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA2[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H2[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI0H2[j][k][i] = true;
                    WLI0H2[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL1->H2
#pragma omp parallel for private(i, j, k) shared(HLA2, ALI1H2, WLI1H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA2[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H2[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI1H2[j][k][i] = true;
                    WLI1H2[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL2->H2
#pragma omp parallel for private(i, j, k) shared(HLA2, ALI2H2, WLI2H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA2[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H2[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI2H2[j][k][i] = true;
                    WLI2H2[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL0->H3
#pragma omp parallel for private(i, j, k) shared(HLA3, ALI0H3, WLI0H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA3[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H3[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI0H3[j][k][i] = true;
                    WLI0H3[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL1->H3
#pragma omp parallel for private(i, j, k) shared(HLA3, ALI1H3, WLI1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA3[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H3[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI1H3[j][k][i] = true;
                    WLI1H3[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL2->H3
#pragma omp parallel for private(i, j, k) shared(HLA3, ALI2H3, WLI2H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA3[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }

        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H3[j][k][i] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALI2H3[j][k][i] = true;
                    WLI2H3[j][k][i] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }

    // IL0->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI0OL, WLI0OL)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI0OL[i][j][k][l] == true) // Node is already active
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                    { // We create a new connection
                        ALI0OL[i][j][k][l] = true;
                        WLI0OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                    }
                }
            }
        }
    }

    // IL1->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI1OL, WLI1OL)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI1OL[i][j][k][l] == true) // Node is already active
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                    { // We create a new connection
                        ALI1OL[i][j][k][l] = true;
                        WLI1OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                    }
                }
            }
        }
    }

    // IL2->OL
#pragma omp parallel for private(i, j, k, l) shared(ALI2OL, WLI2OL)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI2OL[i][j][k][l] == true) // Node is already active
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                    { // We create a new connection
                        ALI2OL[i][j][k][l] = true;
                        WLI2OL[i][j][k][l] = ((float)(rand() % 100000) / 50000) - 1;
                    }
                }
            }
        }
    }

    // H1->H2
#pragma omp parallel for private(i, j) shared(HLA1, HLA2, ALH1H2, WLH1H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (HLA2[j] == 0)
            { // Can only create a connection if the node is active
                continue;
            }
            if (ALH1H2[i][j] == true) // Node is already active
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
            { // We create a new connection
                ALH1H2[i][j] = true;
                WLH1H2[i][j] = ((float)(rand() % 100000) / 50000) - 1;
            }
        }
    }

    // H1->H3
#pragma omp parallel for private(i, j) shared(HLA1, HLA3, ALH1H3, WLH1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (HLA3[j] == 0)
            { // Can only create a connection if the node is active
                continue;
            }
            if (ALH1H3[i][j] == true) // Node is already active
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
            { // We create a new connection
                ALH1H3[i][j] = true;
                WLH1H3[i][j] = ((float)(rand() % 100000) / 50000) - 1;
            }
        }
    }

    // H2->H3
#pragma omp parallel for private(i, j) shared(HLA2, HLA3, ALH2H3, WLH2H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA2[i] == 0)
        { // Can only create a connection if the node is active
            continue;
        }
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (HLA3[j] == 0)
            { // Can only create a connection if the node is active
                continue;
            }
            if (ALH2H3[i][j] == true) // Node is already active
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
            { // We create a new connection
                ALH2H3[i][j] = true;
                WLH2H3[i][j] = ((float)(rand() % 100000) / 50000) - 1;
            }
        }
    }
}

void GenomeCVRP::setFitness(int value)
{
    fitness = value;
}
int GenomeCVRP::getFitness()
{
    return fitness;
}
// TODO Create function to determine an individuals fitness (Mainly get from CPLEX)

//-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****

PopulationCVRP::PopulationCVRP()
{
    population[0] = GenomeCVRP();
    population[0].mutateGenomeWeights();
    population[0].mutateGenomeNodeStructure();
    population[0].mutateGenomeConnectionStructure();
    cout << "\nCreating Population: " << population[0].fitness << "\n";
    getCVRP(15, "R", "R", 4);
    runCPLEXGenome();
}
void PopulationCVRP::getCVRP(int nCostumers, string depotLocation, string customerDistribution, int demandDistribution)
{
    tuple<vector<vector<int>>, int> instance = generateCVRPInstance(nCostumers, depotLocation, customerDistribution, demandDistribution); // generate an instance of the problem
    vertexMatrix = get<0>(instance);
    capacity = get<1>(instance);
    costMatrix = calculateEdgeCost(&vertexMatrix);
}
void PopulationCVRP::getCVRP(string filepath)
{
    tuple<vector<vector<int>>, int> instance = readCVRP(filepath);
    vertexMatrix = get<0>(instance);
    capacity = get<1>(instance);
    costMatrix = calculateEdgeCost(&vertexMatrix);
}

void PopulationCVRP::runCPLEXGenome()
{
    for (GenomeCVRP g : population)
    {
        cout << "LOLOLO\n"
             << flush;
        cout << time(nullptr) << endl
             << flush;
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, true, g);
    }
}

// TODO Create function to calculate the distance between two individuals (species representative and current individual)

// TODO Create function that will run the cplex with the standar CPLEX branching scheme

// TODO Create function that reports and log what happened in the current generation (one log file is better to use later)

// TODO Create function that will generate an ofspring from two individuals

// TODO Create function that will eliminate x% of the worst individuals of the population

// TODO Create a functional branchingWithGenome

CVRPCallback::CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                           const NumVarMatrix &_edgeUsage, vector<vector<int>> &_costMatrix, vector<vector<int>> &_coordinates, GenomeCVRP &_genome) : edgeUsage(_edgeUsage)
{
    Q = capacity;
    N = n;
    demandVector = demands;
    genome = _genome;
    M1 = twoDimensionVectorCreator(50, 50, (float)0);
    coordinates = twoDimensionVectorCreator(3, 50, (float)0);
    misc = twoDimensionVectorCreator(1, 25, (float)0);
    // Create the first part of M1 (i.e. the distance(cost) between each node)
    //(This is a half constant input for the genome)
    for (int i = 0; i < N; ++i)
    {
        for (int j = (i + 1); j < N; ++j)
        {
            // M1[i][j] = (float)_costMatrix->at(i).at(j) / 1500; // Normalize by the max distance = (2*1000^2)^0.5=1414~1500
            M1[i][j] = (float)_costMatrix[i][j] / 1500; // Normalize by the max distance = (2*1000^2)^0.5=1414~1500
        }
    }

    // Create the coordinates (This is a constant input for the genome)
    for (int i = 0; i < N; ++i)
    {
        coordinates[0][i] = (float)_coordinates[0][i] / 1000;
        coordinates[1][i] = (float)_coordinates[1][i] / 1000;
        coordinates[2][i] = (float)_coordinates[2][i] / capacity;
    }
}

inline void
CVRPCallback::connectedComponents(const IloCplex::Callback::Context &context) const
{
    // cutCalls++;
    vector<int> inSet(N, 0); // If 0 the customer is yet to be added to a connected component set
    inSet[0] = 1;
    int currentNode;
    int setIndex = 0;

    vector<vector<int>> connectedComponentsMatrix;

    // Find all sets of connected components
    for (IloInt i = 1; i < N; i++)
    {
        if (inSet[i] == 1)
        {
            continue;
        }
        vector<int> connectedSet{int(i)};
        inSet[i] = 1;
        setIndex = 0;
        while (setIndex < connectedSet.size())
        {
            currentNode = connectedSet[setIndex];
            // We need to find the neighbours of each vertex
            for (int a = 1; a < N; a++)
            { // We go through the whole line to find the neighbours of current Node
                IloNum const s = context.getCandidatePoint(edgeUsage[currentNode][a]);

                if (s == 0)
                {
                    continue;
                }
                // if not 0 they are next to each other in the route
                if (inSet[a] == 1)
                {
                    continue;
                }             // Not already in the set
                inSet[a] = 1; // Mark as in the set
                connectedSet.push_back(a);
            }
            // We have found all neighbours of the current Node and move on to the next
            setIndex++;
        }
        connectedComponentsMatrix.push_back(connectedSet);
    }

    // Now that we have found all connected components, we will check for the rounded capacity innequalities
    for (vector<int> connectedSet : connectedComponentsMatrix)
    {
        int totalSetDemand = 0;
        int totalReciprocalDemand = 0;
        for (int i : connectedSet)
        {
            totalSetDemand += demandVector[i];
        }
        vector<int> reciprocal((N - 1 - connectedSet.size()), 0);
        bool inSet = false;
        int index = 0;
        for (int i = 1; i < N; i++)
        {
            for (int j : connectedSet)
            {
                if (j == i)
                {
                    inSet = true;
                    break;
                }
            }
            if (inSet == false)
            {
                reciprocal[index] = i;
                totalReciprocalDemand += demandVector[i];
                index++;
            }
            inSet = false;
        }

        int ks = ceil((float)totalSetDemand / (float)Q);
        int candidateConnectingEdges = 0;                       // We check if the current candidate solution satissfies this property
        IloNumExpr connectingEdges = IloExpr(context.getEnv()); // These are the edges leaving the connected set
        for (int i : connectedSet)
        {
            connectingEdges += edgeUsage[i][0];
            candidateConnectingEdges += context.getCandidatePoint(edgeUsage[i][0]);
            for (int j : reciprocal)
            {
                connectingEdges += edgeUsage[i][j];
                candidateConnectingEdges += context.getCandidatePoint(edgeUsage[i][j]);
            }
        }
        if (candidateConnectingEdges < 2 * ks)
        {
            context.rejectCandidate(connectingEdges >= 2 * ks); // As any vehicle must enter and leave the set it is times 2
        }
        connectingEdges.end();

        // Now we do the same thing for the reciprocal set
        ks = ceil((float)totalReciprocalDemand / (float)Q);
        candidateConnectingEdges = 0;                                     // We check if the current candidate solution satissfies this property
        IloNumExpr connectingEdgesReciprocal = IloExpr(context.getEnv()); // These are the edges leaving the connected set
        for (int i : reciprocal)
        {
            connectingEdgesReciprocal += edgeUsage[i][0];
            candidateConnectingEdges += context.getCandidatePoint(edgeUsage[i][0]);
            for (int j : connectedSet)
            {
                connectingEdgesReciprocal += edgeUsage[i][j];
                candidateConnectingEdges += context.getCandidatePoint(edgeUsage[i][j]);
            }
        }
        if (candidateConnectingEdges < 2 * ks)
        {
            context.rejectCandidate(connectingEdgesReciprocal >= 2 * ks); // As any vehicle must enter and leave the set it is times 2
        }
        connectingEdgesReciprocal.end();
    }
}

inline void
CVRPCallback::branchingWithGenome(const IloCplex::Callback::Context &context) const
{
    // Get the current relaxation.
    // The function not only fetches the objective value but also makes sure
    // the node lp is solved and returns the node lp's status. That status can
    // be used to identify numerical issues or similar
    IloCplex::CplexStatus status = context.getRelaxationStatus(0);
    double obj = context.getRelaxationObjective();

    // Only branch if the current node relaxation could be solved to
    // optimality.
    // If there was any sort of trouble then don't do anything and thus let
    // CPLEX decide how to cope with that.
    if (status != IloCplex::Optimal &&
        status != IloCplex::OptimalInfeas)
    {
        return;
    }

    {
        mtxCVRP.lock(); // Place in the lock everything that is shared between the nodes
        branches++;
        branches++;
        for (IloInt i = 1; i < N; ++i)
        {
            for (IloInt j = 0; j < i; j++)
            {
                M1[i][j] = context.getRelaxationPoint(edgeUsage[i][j]);
            }
        }

        CPXLONG upChild, downChild;
        double const up = 1;
        double const down = 0;

        vector<int> branchVarIJ = genome.feedForwardFirstTry(&M1, &coordinates, &misc);
        cout << "\ni j " << flush;

        cout << "\ni: " << branchVarIJ[0] << flush;
        cout << "\nj: " << branchVarIJ[1] << flush;
        IloNumVar branchVar = edgeUsage[branchVarIJ[0]][branchVarIJ[1]];
        // Create UP branch (branchVar >= up)
        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        mtxCVRP.unlock();
    }
}

int CVRPCallback::getCalls() const
{
    return cutCalls;
}
int CVRPCallback::getBranches() const
{
    return branches;
}

// This is the function that we have to implement and that CPLEX will call
// during the solution process at the places that we asked for.
// virtual void CVRPCallback::invoke(const IloCplex::Callback::Context &context) ILO_OVERRIDE;

// This is the method that we have to implement to fulfill the
// generic callback contract. CPLEX will call this method during
// the solution process at the places that we asked for.
void CVRPCallback::invoke(const IloCplex::Callback::Context &context)
{
    if (context.inCandidate())
    {
        connectedComponents(context);
    }
    else if (context.inBranching())
    {
        branchingWithGenome(context);
    }
}
CVRPCallback::~CVRPCallback()
{
}

NumVarMatrix populate(IloModel *model, NumVarMatrix edgeUsage, vector<vector<int>> *vertexMatrix, vector<vector<int>> *costMatrix, int *capacity)
{
    int n = vertexMatrix->size();
    int sumOfDemands = 0;
    for (int i = 1; i < n; i++) // Get the demands for each customer
    {
        sumOfDemands += vertexMatrix->at(i).at(2);
    }
    int kMin = ::ceil((float)sumOfDemands / *capacity); // At least 1, extra if the demand is larger than the caacity

    IloEnv env = model->getEnv(); // get the environment
    IloObjective obj = IloMinimize(env);

    // Create all the edge variables
    IloNumVarArray EedgeUsage = IloNumVarArray(env, n, 0, 2, ILOINT); // Edges to the depot may be travelled twice, if the route is one costumer
    edgeUsage[0] = EedgeUsage;
    // Edges to the depot may be travelled twice, if the route is one costumer
    // edgeUsage[0] = IloNumVarArray(env, n, 0, 2, ILOINT); // Edges to the depot may be travelled twice, if the route is one costumer
    edgeUsage[0][0].setBounds(0, 0); // Cant travel from the depot to itself

    for (int i = 1; i < n; i++)
    {
        edgeUsage[i] = IloNumVarArray(env, n, 0, 1, ILOINT); // All other edges should be traversed at most once
        edgeUsage[i][i].setBounds(0, 0);                     // Cant travel from vertex i to vertex i
        edgeUsage[i][0].setBounds(0, 2);                     // Edges from the depot can be traveled twice
    }

    // Each vertex (except the depot) must have degree 2
    for (IloInt i = 1; i < n; ++i)
    {
        model->add(IloSum(edgeUsage[i]) == 2); // i.e. for each row the sum of edge Usage must be two
    }
    model->add(IloSum(edgeUsage[0]) >= 2 * kMin); // The depot must have degree of at least 2 kMin

    IloExpr v(env);
    for (IloInt i = 0; i < n; ++i)
    { // first column must be larger than 2*Kmin
        v += edgeUsage[i][0];
    }
    model->add(v >= 2 * kMin);
    v.end();

    for (IloInt j = 1; j < n; ++j)
    {
        IloExpr v(env);
        for (IloInt i = 0; i < n; ++i)
        { // all columns muust be equal to two as well
            v += edgeUsage[i][j];
        }
        model->add(v == 2);
        v.end();
    }

    // Making the edges usage matrix t be symmmetric
    for (IloInt j = 1; j < n; j++)
    {
        for (IloInt i = (j + 1); i < n; i++)
        {
            IloExpr v(env);
            v += edgeUsage[i][j];
            v -= edgeUsage[j][i];
            model->add(v == 0); // Forcing symmetry
            v.end();
        }
    }

    // Create the objective function i.e. Cij*Xij
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            obj.setLinearCoef(edgeUsage[i][j], costMatrix->at(i).at(j)); // i.e. Xij * Cij
        }
    }

    model->add(obj);
    return edgeUsage;
}

tuple<vector<vector<int>>, int> createAndRunCPLEXInstance(vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity, bool training, GenomeCVRP genome)
{
    cout << time(nullptr) << endl
         << flush;

    IloEnv env;
    IloModel model(env);
    int n = vertexMatrix.size();
    NumVarMatrix edgeUsage(env, n);
    vector<int> demands(n, 0);

    for (int i = 1; i < n; i++) // Get the demands for each customer
    {
        demands[i] = vertexMatrix[i][2];
    }

    edgeUsage = populate(&model, edgeUsage, &vertexMatrix, &costMatrix, &capacity);
    IloCplex cplex(model);
    cplex.setParam(IloCplex::Param::TimeLimit, 600);
    cplex.setParam(IloCplex::Param::Threads, 12);

    // Custom Cuts and Branch in one Custom Generic Callback
    CPXLONG contextMask = 0;
    contextMask |= IloCplex::Callback::Context::Id::Candidate;
    cout << "1" << flush;
    // If true we will use the custom branching scheme with the genome as the AI
    if (training = true)
    {
        contextMask |= IloCplex::Callback::Context::Id::Branching;
    }
    cout << "1" << flush;

    CVRPCallback cgc(capacity, n, demands, edgeUsage, costMatrix, vertexMatrix, genome);
    cout << "1" << flush;

    cplex.use(&cgc, contextMask);

    // These are the CUTS that are standard in CPLEX,
    // we can remove them to force CPLEX to use the custom Cuts
    cplex.setParam(IloCplex::Param::MIP::Strategy::HeuristicFreq, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::MIRCut, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::Implied, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::Gomory, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::FlowCovers, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::PathCut, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::LiftProj, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::ZeroHalfCut, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::Cliques, -1);
    cplex.setParam(IloCplex::Param::MIP::Cuts::Covers, -1);
    cplex.setOut(env.getNullStream());
    // Optimize the problem and obtain solution.
    if (!cplex.solve())
    {
        env.error() << "Failed to optimize LP" << endl;
    }

    NumVarMatrix vals(env);
    env.out() << "Solution status = " << cplex.getStatus() << endl;
    env.out() << "Solution value  = " << cplex.getObjValue() << endl;
    int value;
    vector<vector<int>> edgeUsageSolution(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            value = cplex.getValue(edgeUsage[i][j]);
            edgeUsageSolution.at(i).at(j) = value;
        }
    }
    int cost = cplex.getObjValue();

    cout << "Custom branches created: " << cgc.getBranches() << endl;

    env.end();
    return make_tuple(edgeUsageSolution, cost);
}

int main()
{

    srand((unsigned int)time(NULL));
    GenomeCVRP g0;
    // cout << "Sleeping\n" << flush;
    // sleep(5);
    PopulationCVRP p0;
    vector<vector<float>> M1 = twoDimensionVectorCreator(500, 500, (float)1);
    vector<vector<float>> coordinates = twoDimensionVectorCreator(3, 500, (float)1);
    vector<vector<float>> misc = twoDimensionVectorCreator(1, 50, (float)1);
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

    // g0.feedForwardFirstTry(&M1, &coordinates, &misc);
    // g0.mutateGenomeWeights();
    // g0.mutateGenomeNodeStructure();
    // g0.mutateGenomeConnectionStructure();

    return 0;
}