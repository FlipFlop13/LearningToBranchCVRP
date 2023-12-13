#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <vector>
#include <list>
#include <tuple>
#include <map>
#include <mutex>
#include <ctime>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>
// #include "omp.h"
#include <omp.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <bits/stdc++.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/cplex.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/ilocplex.h>
#include <chrono>

#include "./utilities.h"
#include "./CVRPGrapher.h"
// #include "./branchAndCut.h"
using namespace std;

std::mutex mtxCVRP; // mutex for critical section
std::mutex mtxInvoke; // mutex for critical section

#define MUTATEWEIGHTSPROB 0.8       // The probability that this genome will have its weights mutated, if they are mutatet each weight will be mutate with p as follows
#define PERTURBEPROB 1.0            // Probability that the weight of a gene is perturbed (in reality this is PERTURBEPROB-MUTATEPROB)
#define MUTATEPROB 0.1              // or that a complete new value is given for the weight
#define NEWNODEPROB 0.001           // Probability that a new node is added to split an existing edge
#define NEWCONNECTIONPROB 0.05      // Probability that a new edge is added between two existing nodes
#define REMOVECONNECTIONPROB 0.0125 // The probability that an existing connection gets removed (In equilibrium the percentage of active connections is  (NEWCONNECTIONPROB/(NEWCONNECTIONPROB+REMOVECONNECTIONPROB)))
#define C1 1                        //
#define C2 1                        //
#define C3 1                        //
#define nSpeciesThreshold 10        // We never let the number of species be higher than 10, if this happens we decrease the likeness thraeshold
#define DIRECTORY "./NEAT/run16/"
#define WEIGHTMUTATIONRANGE 0.2 // the value range that the weight will be perturbed with
#define WEIGHTMUTATIONMEAN 0.0  // the mean perturbation
#define SIGMA 4.9               // the sigma value for the sigmoid activation function

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

public:
    mutable int nnTime = 0;

    int maxNodesHiddenLayer = 2000;
    int maxInputSize = 15;
    int s = 5;
    int bias = 1;
    float startingValue = 0.01;
    bool TEMPBOOL = 0;
    int activeConnections = 0;
    int randID = rand();
    vector<int> fitness = oneDimensionVectorCreator(3, 0);
    float fitnessFloat = (float)(rand() % 100000) / 100000;

    // save the node orders of the last instance
    mutable map<int, tuple<int, int, int, int>> searchTreeParentToChildren;
    mutable map<int, int> searchTreeChildToParent;
    int lastID;
    vector<vector<int>> result; // We save the result of the last instance

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
    // This is the bias weights to the hidden layer 1
    vector<float> BH1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

    // IL->HL2
    //  These are the activation layers between the input and the second hidden layer
    vector<vector<vector<bool>>> ALI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H2 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H2 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H2 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);
    // This is the bias weights to the hidden layer 2
    vector<float> BH2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

    // IL->HL3
    //  These are the activation layers between the input and the third hidden layer
    vector<vector<vector<bool>>> ALI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (bool)TEMPBOOL);
    vector<vector<vector<bool>>> ALI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (bool)TEMPBOOL);
    // These are the weight layers between the input and the hidden layers
    vector<vector<vector<float>>> WLI0H3 = threeDimensionVectorCreator(maxInputSize, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI1H3 = threeDimensionVectorCreator(3, maxInputSize, maxNodesHiddenLayer, (float)startingValue);
    vector<vector<vector<float>>> WLI2H3 = threeDimensionVectorCreator(1, s, maxNodesHiddenLayer, (float)startingValue);
    // This is the bias weights to the hidden layer 3
    vector<float> BH3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)startingValue);

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
    // This is the bias weights to the output
    vector<vector<float>> BO = twoDimensionVectorCreator(maxInputSize, maxInputSize, (float)startingValue);

    //------------*********Functions***********------------------
    GenomeCVRP();
    tuple<int, int> feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<float>> *coordinates, vector<vector<float>> *misc, vector<vector<bool>> *splittedBefore) const;
    void mutateGenomeWeights();
    void mutateGenomeNodeStructure();
    void mutateGenomeConnectionStructure();
    void removeGenomeConnections();
    void countActiveConnections();
    void setFitness(vector<int> value);
    vector<int> getFitness();
    void setFloatFitnessG(float fitnessFloatValue);
    float getFloatFitness();
    void inheritParentData(GenomeCVRP &g);

    template <class Archive>
    void serialize(Archive &a, const unsigned version);

    void clearGenomeRAM();
};
class PopulationCVRP
{

private:
    int populationSize = 50;
    int nCostumers = 10; // startin value of the number of costumers for the evolution loop

    GenomeCVRP defaultCPLEXPlaceholder;
    GenomeCVRP minInfeasibleCPLEXPlaceholder;
    GenomeCVRP maxInfeasibleCPLEXPlaceholder;
    GenomeCVRP pseudoCPLEXPlaceholder;
    GenomeCVRP strongCPLEXPlaceholder;
    GenomeCVRP pseudoReducedCPLEXPlaceholder;
    GenomeCVRP heuristicShortestPlaceholder;
    GenomeCVRP heuristicLongestPlaceholder;
    GenomeCVRP heuristicRandomPlaceholder;
    string logFile = (string)DIRECTORY + "log.txt";

public:
    vector<vector<GenomeCVRP>> population;
    int generation = 0;
    vector<vector<int>> vertexMatrix; // This is the instance of CVRP for whihc the current generation is run on
    int capacity;
    vector<vector<int>> costMatrix;

    // This vector contains all the ids of the genomes in the population, used for saving
    vector<vector<int>> genomeIDs;

    PopulationCVRP();

    void log(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers, bool branched);
    void logTable(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers);
    void logSemiColonSeparated(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers);
    void logBestSplitOrder(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers);

    void setLogFile(string filepath);
    void getCVRP(int nCostumers, string depotLocation, string customerDistribution, int demandDistribution);
    void getCVRP(string filepath);
    void initializePopulation();
    bool runCPLEXGenome();
    void reorganizeSpecies();
    void decimate(float percentage = 0.1);
    void decimatePreserveSpecies(float percentage = 0.1);
    void eliminateSpeciesAndMakeNew();
    void reproducePopulation();
    void mutatePopulation();
    void evolutionLoop();
    void saveGenomesAndFreeRAM();
    void saveGenomes();
    void loadAllGenomes();
    void orderSpecies();
    void setFloatFitnessP();
    void updateGenomeIDs();
    void IDToPopulation();
    GenomeCVRP reproduce(GenomeCVRP &g0, GenomeCVRP &g1);

    template <class Archive>
    void serialize(Archive &a, const unsigned version);
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

    // Variable representing the type of branching that needs to be done 1 is heuristic and 2 genome (0 is standard cplex)
    int T;

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

    // This map will keep track of the structure of the cplex tree, key is the node and value the parent node
    mutable map<int, int> nodeStructure;
    // This map keps track of how each node is split, it maps the parent id to a tuple of child1ID child2ID branched i and j
    mutable map<int, tuple<int, int, int, int>> nodeStructureParentToChilds;
    // This map will keep track of wich variables (edge) was split in this node
    mutable map<int, tuple<int, int>> nodeSplit;
    mutable CPXLONG currentNodeID;
    mutable CPXLONG lastNodeID;
    mutable int ioTime = 0;
    mutable int nnTime = 0;
    mutable int invokeTime = 0;

public:
    CVRPCallback();
    // Constructor with data.
    CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                 const NumVarMatrix &_edgeUsage, const vector<vector<int>> &_costMatrix, const vector<vector<int>> &_coordinates, const GenomeCVRP &_genome, const int genomeBool);

    inline void 
    branchingWithCPLEX() const;
    inline void
    connectedComponents(const IloCplex::Callback::Context &context) const;
    inline void
    branchingWithGenome(const IloCplex::Callback::Context &context) const;
    inline void
    branchingWithShortestHeuristic(const IloCplex::Callback::Context &context) const;
    inline void
    branchingWithLongestHeuristic(const IloCplex::Callback::Context &context) const;
    inline void
    branchingWithRandomHeuristic(const IloCplex::Callback::Context &context) const;
    tuple<int, int> findClosestNotSplit(vector<vector<bool>> *splittedBefore) const;
    tuple<int, int> findFurthestNotSplit(vector<vector<bool>> *splittedBefore) const;
    tuple<int, int> findRandomNotSplit(vector<vector<bool>> *splittedBefore) const;

    int getCalls() const;
    int getBranches() const;
    virtual void invoke(const IloCplex::Callback::Context &context) ILO_OVERRIDE;
    virtual ~CVRPCallback();
    vector<vector<bool>> getEdgeBranches() const;

    int getLastNodeID() const;
    map<int, tuple<int, int, int, int>> getNodeStructureParentToChilds() const;
    map<int, int> getNodeStructureChildToParent() const;

    void printTimes();
};

tuple<vector<vector<int>>, int> createAndRunCPLEXInstance(const vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity, int runType, GenomeCVRP &genome, const int timeout);
float calculateLikeness(GenomeCVRP &g0, GenomeCVRP &g1);
void saveGenomeInstance(GenomeCVRP &genome);
void loadGenomeInstance(GenomeCVRP &genome);
void savePopulation(PopulationCVRP &population);
void loadPopulation(PopulationCVRP &population);

//------------------------------GenomeCVRP---------------------------------------------------
GenomeCVRP::GenomeCVRP()
{
    cout << "";
}

template <class Archive>
void GenomeCVRP::serialize(Archive &ar, const unsigned int file_version)
{
    // boost::serialization::split_member(ar, *this, file_version);
    ar & HLA1 & HLA2 & HLA3 & BH1 & BH2 & BH3 & BO & ALI0OL & ALI1OL & ALI2OL & WLI0OL & WLI1OL & WLI2OL & ALI0H1 & ALI1H1 & ALI2H1 & WLI0H1 & WLI1H1 & WLI2H1 & ALI0H2 & ALI1H2 & ALI2H2 & WLI0H2 & WLI1H2 & WLI2H2 & ALI0H3 & ALI1H3 & ALI2H3 & WLI0H3 & WLI1H3 & WLI2H3 & ALH1H2 & ALH1H3 & ALH2H3 & WLH1H2 & WLH1H3 & WLH2H3 & ALH1OL & ALH2OL & ALH3OL & WLH1OL & WLH2OL & WLH3OL;
}
void saveGenomeInstance(GenomeCVRP &genome)
{

    string filename = string(DIRECTORY) + "data/Genome" + to_string(genome.randID) + ".dat";
    std::ofstream outfile(filename, std::ofstream::binary);
    // boost::archive::text_oarchive archive(outfile);
    boost::archive::binary_oarchive archive(outfile, boost::archive::no_header);
    archive & genome;
}

void loadGenomeInstance(GenomeCVRP &genome)
{

    string filename = string(DIRECTORY) + "data/Genome" + to_string(genome.randID) + ".dat";
    std::ifstream infile(filename, std::ifstream::binary);
    // boost::archive::text_iarchive archive(infile);
    boost::archive::binary_iarchive archive(infile, boost::archive::no_header);
    archive & genome;
}

tuple<int, int> GenomeCVRP::feedForwardFirstTry(vector<vector<float>> *M1, vector<vector<float>> *coordinates, vector<vector<float>> *misc, vector<vector<bool>> *splittedBefore) const
{

    // Reset hidden layer values
    // step 1, calculate the values of the first hidden layer
    int i, j, k, l;
    // TIMER
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ios_base::sync_with_stdio(false);
    HLV1 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    HLV2 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);
    HLV3 = oneDimensionVectorCreator(maxNodesHiddenLayer, (float)0);

    // Step 1.1 M1->H1
    auto startTime = chrono::high_resolution_clock::now();
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
    // auto stopTime = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
    // cout << "DurationP: " << duration.count() << "\n" << flush;
    // startTime = chrono::high_resolution_clock::now();
    // for (k = 0; k < maxNodesHiddenLayer; ++k)
    // {
    //     for (i = 0; i < maxInputSize; ++i)
    //     {
    //         for (j = 0; j < maxInputSize; ++j)
    //         {
    //             HLV1[k] += M1->at(i).at(j) * ALI0H1[i][j][k] * WLI0H1[i][j][k];
    //         }
    //     }
    // }
    // stopTime = chrono::high_resolution_clock::now();
    // duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
    // cout << "DurationS: " << duration.count() << "\n" << flush;
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

    // Add the bias and calculate the sigmoid of the layer
#pragma omp parallel for private(i) shared(BH1, HLA1, HLV1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        HLV1[i] += HLA1[i] * BH1[i];
        HLV1[i] = sigmoid(HLV1[i], SIGMA);
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
    // Add the bias and calculate the sigmoid of the layer
#pragma omp parallel for private(i) shared(BH2, HLA2, HLV2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        HLV2[i] += HLA2[i] * BH2[i];
        HLV2[i] = sigmoid(HLV2[i], SIGMA);
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

    // Step 3.5 H2->H3
#pragma omp parallel for private(i, j) shared(HLV3, HLV2, ALH1H3, WLH1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            HLV3[i] += HLV2[j] * ALH1H3[i][j] * WLH1H3[i][j];
        }
    }

    // Add the bias
    // Add the bias and calculate the sigmoid of the layer
#pragma omp parallel for private(i) shared(BH3, HLA3, HLV3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        HLV3[i] += HLA3[i] * BH3[i];
        HLV3[i] = sigmoid(HLV3[i], SIGMA);
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

    // Add the bias to the output layer and perform the sigmoid
#pragma omp parallel for private(i, j) shared(BO, OUTPUT2D)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            OUTPUT2D[i][j] += BO[i][j];
            OUTPUT2D[i][j] = sigmoid(OUTPUT2D[i][j], SIGMA);
        }
    }

    // Only consider for a split if the pair has not yet been splitted and if it is in the range of the current problem (note that is not the same as the genome)
    int currentProblemSize = splittedBefore->size();
#pragma omp parallel for private(i, j) shared(BO, OUTPUT2D)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            if (!(i < currentProblemSize) || !(j < currentProblemSize))
            { // If the current output is outside the current instance scope
                OUTPUT2D[i][j] = 0;
                continue;
            }
            if (!(splittedBefore->at(i).at(j)))
            {
                continue;
            }
            OUTPUT2D[i][j] = 0;
        }
    }

    // Step 6 convert 2d to a vector of i and j
    tuple<int, int> OUTPUT1D;
    float maxValue = EPSU;
#pragma omp parallel for private(i, j) shared(OUTPUT2D, OUTPUT1D)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            if (OUTPUT2D[i][j] > maxValue)
            {

                maxValue = OUTPUT2D[i][j];
                OUTPUT1D = make_tuple(i, j);
            }
        }
    }

    gettimeofday(&end, NULL);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec -
                                start.tv_usec)) *
                 1e-6;

    // stopTime = chrono::high_resolution_clock::now();
    // duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
    auto stopTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
    nnTime += duration.count();
    // cout << "Duration: " << duration.count() << "\n" << flush;
    return OUTPUT1D;
}

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
                            WLI0OL[i][j][k][l] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                            WLI1OL[i][j][k][l] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                            WLI2OL[i][j][k][l] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI0H1[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI1H1[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI2H1[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
                    }
                }
            }
        }
    }

    // BH1

#pragma omp parallel for private(i) shared(HLA1, BH1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA1[i] == 0)
        {
            continue;
        }
        randFloat = (float)(rand() % 100000) / 100000;
        if (randFloat <= MUTATEPROB)
        {
            BH1[i] = ((float)(rand() % 100000) / 50000) - 1;
            continue;
        }
        if (randFloat < PERTURBEPROB)
        {
            BH1[i] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI0H2[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI1H2[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI2H2[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
                    }
                }
            }
        }
    }

// BH2
#pragma omp parallel for private(i) shared(HLA2, BH2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA2[i] == 0)
        {
            continue;
        }
        randFloat = (float)(rand() % 100000) / 100000;
        if (randFloat <= MUTATEPROB)
        {
            BH2[i] = ((float)(rand() % 100000) / 50000) - 1;
            continue;
        }
        if (randFloat < PERTURBEPROB)
        {
            BH2[i] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI0H3[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI1H3[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLI2H3[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
                    }
                }
            }
        }
    }

// BH3
#pragma omp parallel for private(i) shared(HLA3, BH3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        if (HLA3[i] == 0)
        {
            continue;
        }
        randFloat = (float)(rand() % 100000) / 100000;
        if (randFloat <= MUTATEPROB)
        {
            BH3[i] = ((float)(rand() % 100000) / 50000) - 1;
            continue;
        }
        if (randFloat < PERTURBEPROB)
        {
            BH3[i] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                    WLH1H2[i][j] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                    WLH1H3[i][j] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                    WLH2H3[i][j] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLH1OL[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLH2OL[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
                        WLH3OL[i][j][k] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
                    }
                }
            }
        }
    }

// BO
#pragma omp parallel for private(i, j) shared(BO)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            randFloat = (float)(rand() % 100000) / 100000;
            if (randFloat <= MUTATEPROB)
            {
                // If we mutate we get a complete new random value a random float value between -1 and 1
                BO[i][j] = ((float)(rand() % 100000) / 50000) - 1;
                continue;
            }
            if (randFloat < PERTURBEPROB)
            {
                // If we perturbe by a value between -0.1 & 0.1
                BO[i][j] += randomUniformFloatGenerator(WEIGHTMUTATIONRANGE, WEIGHTMUTATIONMEAN);
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
    int i, j, k, l, m, newNode;
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
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALI0H3, WLI0H3, ALI0H1, WLI0H1, ALI0H2, WLI0H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
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
                        l1 = l;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node k is not yet active
                        l2 = l;
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
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALI1H3, WLI1H3, ALI1H1, WLI1H1, ALI1H2, WLI1H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
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
                    { // node l is not yet active
                        l1 = l;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node l is not yet active
                        l2 = l;
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
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALI2H3, WLI2H3, ALI2H1, WLI2H1, ALI2H2, WLI2H2, ALH1H3, WLH1H3, ALH2H3, WLH2H3, HLA1, HLA2)
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
                    { // node l is not yet active
                        l1 = l;
                        freeNode1 = true;
                    }
                    if (HLA2[l] == false)
                    { // node l is not yet active
                        l2 = l;
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
#pragma omp parallel for private(i, j, k, m, l, l1, l2, l3) shared(ALI0OL, WLI0OL, ALI0H1, WLI0H1, ALI0H2, WLI0H2, ALI0H3, WLI0H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH3OL, HLA1, HLA2, HLA3)
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
                        { // node l is not yet active
                            l1 = l;
                            freeNode1 = true;
                        }
                        if (HLA2[l] == false)
                        { // node l is not yet active
                            l2 = l;
                            freeNode2 = true;
                        }
                        if (HLA3[l] == false)
                        { // node l is not yet active
                            l3 = l;
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
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALI1OL, WLI1OL, ALI1H1, WLI1H1, ALI1H2, WLI1H2, ALI1H3, WLI1H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH3OL, HLA1, HLA2, HLA3)
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
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALI2OL, WLI2OL, ALI2H1, WLI2H1, ALI2H2, WLI2H2, ALI2H3, WLI2H3, ALH1OL, WLH1OL, ALH2OL, WLH2OL, ALH3OL, WLH3OL, HLA1, HLA2, HLA3)
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
                        { // node l is not yet active
                            l1 = l;
                            freeNode1 = true;
                        }
                        if (HLA2[l] == false)
                        { // node l is not yet active
                            l2 = l;
                            freeNode2 = true;
                        }
                        if (HLA3[l] == false)
                        { // node l is not yet active
                            l3 = l;
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
                { // node l is not yet active
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

    //     // HL1->OL

#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALH1OL, WLH1OL, ALH1H2, WLH1H2, ALH2OL, WLH2OL, ALH1H3, WLH1H3, ALH3OL, WLH3OL, HLA2, HLA3)
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
                    { // node l is not yet active
                        l2 = l;
                        freeNode2 = true;
                    }
                    if (HLA3[l] == false)
                    { // node l is not yet active
                        l3 = l;
                        freeNode3 = true;
                    }
                }

                if (freeNode2 & s & freeNode3) // Node can be added in both layers
                {
                    randFloat = (float)(rand() % 100000) / 100000;
                    if (randFloat > 0.5)
                    {

                        HLA2[l2] = true;
                        ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                        ALH1H2[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                        WLH1H2[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH2OL[l2][j][k] = true;
                        WLH2OL[l2][j][k] = (float)1; // and for the second it is 1, effectively not changing the values initially
                    }
                    else
                    {
                        HLA3[l3] = true;
                        ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                        ALH1H3[i][l3] = true;            // The connection is now going to node l in the first hidden layer
                        WLH1H3[i][l3] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                        ALH3OL[l3][j][k] = true;
                        WLH3OL[l3][j][k] = (float)1; // and for the second it is 1, effectively not changing the values initially
                    }
                }
                else if (freeNode2) // New node in the first hidden layer
                {
                    cout << "In second layer\n"
                         << flush;

                    HLA2[l2] = true;
                    ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                    ALH1H2[i][l2] = true;            // The connection is now going to node l in the first hidden layer
                    WLH1H2[i][l2] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH2OL[l2][j][k] = true;
                    WLH2OL[l2][j][k] = (float)1; // and for the second it is 1, effectively not changing the values initially
                }
                else if (freeNode3) // New node in the second hidden layer
                {
                    HLA3[l3] = true;
                    ALH1OL[i][j][k] = false; // This connection no longer exists, as it is split

                    ALH1H3[i][l3] = true;            // The connection is now going to node l in the first hidden layer
                    WLH1H3[i][l3] = WLH1OL[i][j][k]; // The first weight is equal to the previous connection weight

                    ALH3OL[l3][j][k] = true;
                    WLH3OL[l3][j][k] = (float)1; // and for the second it is 1, effectively not changing the values initially
                }
                // Else do nothing as there are no places for new nodes
            }
        }
    }

// HL2->OL
#pragma omp parallel for private(i, j, k, l, l2, l3) shared(ALH2OL, WLH2OL, ALH2H3, WLH2H3, ALH3OL, WLH3OL, HLA3)
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
                    { // node l is not yet active
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
                WLH3OL[l][j][k] = (float)1; // and for the second it is 1, effectively not changing the values initially
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
// H1->OL
#pragma omp parallel for private(i, j, k) shared(HLA1, ALH1OL, WLH1OL)
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
                if (ALH1OL[i][j][k] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALH1OL[i][j][k] = true;
                    WLH1OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }
// H2->OL
#pragma omp parallel for private(i, j, k) shared(HLA2, ALH2OL, WLH2OL)
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
                if (ALH2OL[i][j][k] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALH2OL[i][j][k] = true;
                    WLH2OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }
// H1->OL
#pragma omp parallel for private(i, j, k) shared(HLA3, ALH3OL, WLH3OL)
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
                if (ALH3OL[i][j][k] == true) // Node is already active
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < NEWCONNECTIONPROB)
                { // We create a new connection
                    ALH3OL[i][j][k] = true;
                    WLH3OL[i][j][k] = ((float)(rand() % 100000) / 50000) - 1;
                }
            }
        }
    }
}

void GenomeCVRP::removeGenomeConnections()
{
    int i, j, k, l;
// IL0->H1 & IL0->H2 & IL0->H3
#pragma omp parallel for private(i, j, k) shared(ALI0H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H1[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI0H1[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI0H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H2[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI0H2[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI0H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI0H3[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI0H3[j][k][i] = false;
                }
            }
        }
    }
// IL1->H1 &  IL1->H2 &  IL1->H3
#pragma omp parallel for private(i, j, k) shared(ALI1H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H1[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI1H1[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI1H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H2[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI1H2[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALI1H3[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI1H3[j][k][i] = false;
                }
            }
        }
    }
// IL2->H1 &  IL2->H2 &  IL2->H3
#pragma omp parallel for private(i, j, k) shared(ALI2H1)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H1[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI2H1[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI2H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H2[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI2H2[j][k][i] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALI2H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < 1; ++j)
        {
            for (k = 0; k < s; ++k)
            {
                if (ALI2H3[j][k][i] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALI2H3[j][k][i] = false;
                }
            }
        }
    }
// INPUT->OUTPUT
#pragma omp parallel for private(i, j, k, l) shared(ALI0OL)
    for (i = 0; i < maxInputSize; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI0OL[i][j][k][l] == false) // Connection is already inactive
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                    { // We remove (set as innactive) the connection
                        ALI0OL[i][j][k][l] = false;
                    }
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k, l) shared(ALI1OL)
    for (i = 0; i < 3; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI1OL[i][j][k][l] == false) // Connection is already inactive
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                    { // We remove (set as innactive) the connection
                        ALI1OL[i][j][k][l] = false;
                    }
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k, l) shared(ALI2OL)
    for (i = 0; i < 1; ++i)
    {
        for (j = 0; j < s; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                for (l = 0; l < maxInputSize; ++l)
                {
                    if (ALI2OL[i][j][k][l] == false) // Connection is already inactive
                    {
                        continue;
                    }
                    if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                    { // We remove (set as innactive) the connection
                        ALI2OL[i][j][k][l] = false;
                    }
                }
            }
        }
    }
// H1->OL & H2->OL & H3->OL
#pragma omp parallel for private(i, j, k) shared(ALH1OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH1OL[i][j][k] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALH1OL[i][j][k] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALH2OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH2OL[i][j][k] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALH2OL[i][j][k] = false;
                }
            }
        }
    }
#pragma omp parallel for private(i, j, k) shared(ALH3OL)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxInputSize; ++j)
        {
            for (k = 0; k < maxInputSize; ++k)
            {
                if (ALH3OL[i][j][k] == false) // Connection is already inactive
                {
                    continue;
                }
                if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
                { // We remove (set as innactive) the connection
                    ALH3OL[i][j][k] = false;
                }
            }
        }
    }
    // H1->H2 & H1->H3 & H2->H3
#pragma omp parallel for private(i, j) shared(ALH1H2)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH1H2[i][j] == false) // Connection is already inactive
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
            { // We remove (set as innactive) the connection
                ALH1H2[i][j] = false;
            }
        }
    }
#pragma omp parallel for private(i, j) shared(ALH1H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH1H3[i][j] == false) // Connection is already inactive
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
            { // We remove (set as innactive) the connection
                ALH1H3[i][j] = false;
            }
        }
    }
#pragma omp parallel for private(i, j) shared(ALH2H3)
    for (i = 0; i < maxNodesHiddenLayer; ++i)
    {
        for (j = 0; j < maxNodesHiddenLayer; ++j)
        {
            if (ALH2H3[i][j] == false) // Connection is already inactive
            {
                continue;
            }
            if (((float)rand() / (float)RAND_MAX) < REMOVECONNECTIONPROB)
            { // We remove (set as innactive) the connection
                ALH2H3[i][j] = false;
            }
        }
    }
}

void GenomeCVRP::setFitness(vector<int> fitnessValue)
{
    fitness = fitnessValue;
}
vector<int> GenomeCVRP::getFitness()
{
    return fitness;
}

void GenomeCVRP::setFloatFitnessG(float fitnessFloatValue)
{
    fitnessFloat = fitnessFloatValue;
}
float GenomeCVRP::getFloatFitness()
{
    return fitnessFloat;
}

void GenomeCVRP::countActiveConnections()
{
    int count = 0;

    count += boolVectorCounter(ALI0OL);
    count += boolVectorCounter(ALI1OL);
    count += boolVectorCounter(ALI2OL);
    count += boolVectorCounter(ALI0H1);
    count += boolVectorCounter(ALI1H1);
    count += boolVectorCounter(ALI2H1);
    count += boolVectorCounter(ALI0H2);
    count += boolVectorCounter(ALI1H2);
    count += boolVectorCounter(ALI2H2);
    count += boolVectorCounter(ALI0H3);
    count += boolVectorCounter(ALI1H3);
    count += boolVectorCounter(ALI2H3);
    count += boolVectorCounter(ALH1H2);
    count += boolVectorCounter(ALH1H3);
    count += boolVectorCounter(ALH2H3);
    count += boolVectorCounter(ALH1OL);
    count += boolVectorCounter(ALH2OL);
    count += boolVectorCounter(ALH3OL);

    activeConnections = count;

    return;
}

void GenomeCVRP::inheritParentData(GenomeCVRP &g)
{
    HLA1 = g.HLA1;
    HLA2 = g.HLA2;
    HLA3 = g.HLA3;
    BH1 = g.BH1;
    BH2 = g.BH2;
    BH3 = g.BH3;
    BO = g.BO;
    ALI0OL = g.ALI0OL;
    ALI1OL = g.ALI1OL;
    ALI2OL = g.ALI2OL;
    WLI0OL = g.WLI0OL;
    WLI1OL = g.WLI1OL;
    WLI2OL = g.WLI2OL;
    ALI0H1 = g.ALI0H1;
    ALI1H1 = g.ALI1H1;
    ALI2H1 = g.ALI2H1;
    WLI0H1 = g.WLI0H1;
    WLI1H1 = g.WLI1H1;
    WLI2H1 = g.WLI2H1;
    ALI0H2 = g.ALI0H2;
    ALI1H2 = g.ALI1H2;
    ALI2H2 = g.ALI2H2;
    WLI0H2 = g.WLI0H2;
    WLI1H2 = g.WLI1H2;
    WLI2H2 = g.WLI2H2;
    ALI0H3 = g.ALI0H3;
    ALI1H3 = g.ALI1H3;
    ALI2H3 = g.ALI2H3;
    WLI0H3 = g.WLI0H3;
    WLI1H3 = g.WLI1H3;
    WLI2H3 = g.WLI2H3;
    ALH1H2 = g.ALH1H2;
    ALH1H3 = g.ALH1H3;
    ALH2H3 = g.ALH2H3;
    WLH1H2 = g.WLH1H2;
    WLH1H3 = g.WLH1H3;
    WLH2H3 = g.WLH2H3;
    ALH1OL = g.ALH1OL;
    ALH2OL = g.ALH2OL;
    ALH3OL = g.ALH3OL;
    WLH1OL = g.WLH1OL;
    WLH2OL = g.WLH2OL;
    WLH3OL = g.WLH3OL;
}

void GenomeCVRP::clearGenomeRAM()
{
    clearAndResizeVector(&HLA1);
    clearAndResizeVector(&HLA2);
    clearAndResizeVector(&HLA3);
    clearAndResizeVector(&BH1);
    clearAndResizeVector(&BH2);
    clearAndResizeVector(&BH3);
    clearAndResizeVector(&BO);
    clearAndResizeVector(&ALI0OL);
    clearAndResizeVector(&ALI1OL);
    clearAndResizeVector(&ALI2OL);
    clearAndResizeVector(&WLI0OL);
    clearAndResizeVector(&WLI1OL);
    clearAndResizeVector(&WLI2OL);
    clearAndResizeVector(&ALI0H1);
    clearAndResizeVector(&ALI2H1);
    clearAndResizeVector(&WLI0H1);
    clearAndResizeVector(&WLI1H1);
    clearAndResizeVector(&WLI2H1);
    clearAndResizeVector(&ALI0H2);
    clearAndResizeVector(&ALI1H2);
    clearAndResizeVector(&ALI2H2);
    clearAndResizeVector(&WLI0H2);
    clearAndResizeVector(&WLI1H2);
    clearAndResizeVector(&WLI2H2);
    clearAndResizeVector(&ALI0H3);
    clearAndResizeVector(&ALI1H3);
    clearAndResizeVector(&ALI2H3);
    clearAndResizeVector(&WLI0H3);
    clearAndResizeVector(&WLI1H3);
    clearAndResizeVector(&WLI2H3);
    clearAndResizeVector(&ALH1H2);
    clearAndResizeVector(&ALH1H3);
    clearAndResizeVector(&ALH2H3);
    clearAndResizeVector(&WLH1H2);
    clearAndResizeVector(&WLH1H3);
    clearAndResizeVector(&WLH2H3);
    clearAndResizeVector(&ALH1OL);
    clearAndResizeVector(&ALH2OL);
    clearAndResizeVector(&ALH3OL);
    clearAndResizeVector(&WLH1OL);
    clearAndResizeVector(&WLH2OL);
    clearAndResizeVector(&WLH3OL);

    // HLA1.clear();
    // HLA2.clear();
    // HLA3.clear();
    // ALI0OL.clear();
    // ALI1OL.clear();
    // ALI2OL.clear();
    // WLI0OL.clear();
    // WLI1OL.clear();
    // WLI2OL.clear();
    // ALI0H1.clear();
    // ALI2H1.clear();
    // WLI0H1.clear();
    // WLI1H1.clear();
    // WLI2H1.clear();
    // ALI0H2.clear();
    // ALI1H2.clear();
    // ALI2H2.clear();
    // WLI0H2.clear();
    // WLI1H2.clear();
    // WLI2H2.clear();
    // ALI0H3.clear();
    // ALI1H3.clear();
    // ALI2H3.clear();
    // WLI0H3.clear();
    // WLI1H3.clear();
    // WLI2H3.clear();
    // ALH1H2.clear();
    // ALH1H3.clear();
    // ALH2H3.clear();
    // WLH1H2.clear();
    // WLH1H3.clear();
    // WLH2H3.clear();
    // ALH1OL.clear();
    // ALH2OL.clear();
    // ALH3OL.clear();
    // WLH1OL.clear();
    // WLH2OL.clear();
    // WLH3OL.clear();

    // HLA1 = vector<bool>(0,0);
    // HLA2 = vector<bool>();
    // HLA3 = vector<bool>();
    // ALI0OL = vector<vector<vector<vector<bool>>>>();
    // ALI1OL = vector<vector<vector<vector<bool>>>>();
    // ALI2OL = vector<vector<vector<vector<bool>>>>();
    // WLI0OL = vector<vector<vector<vector<float>>>>();
    // WLI1OL = vector<vector<vector<vector<float>>>>();
    // WLI2OL = vector<vector<vector<vector<float>>>>();
    // ALI0H1 = vector<vector<vector<bool>>>();
    // ALI1H1 = vector<vector<vector<bool>>>();
    // ALI2H1 = vector<vector<vector<bool>>>();
    // WLI0H1 = vector<vector<vector<float>>>();
    // WLI1H1 = vector<vector<vector<float>>>();
    // WLI2H1 = vector<vector<vector<float>>>();
    // ALI0H2 = vector<vector<vector<bool>>>();
    // ALI1H2 = vector<vector<vector<bool>>>();
    // ALI2H2 = vector<vector<vector<bool>>>();
    // WLI0H2 = vector<vector<vector<float>>>();
    // WLI1H2 = vector<vector<vector<float>>>();
    // WLI2H2 = vector<vector<vector<float>>>();
    // ALI0H3 = vector<vector<vector<bool>>>();
    // ALI1H3 = vector<vector<vector<bool>>>();
    // ALI2H3 = vector<vector<vector<bool>>>();
    // WLI0H3 = vector<vector<vector<float>>>();
    // WLI1H3 = vector<vector<vector<float>>>();
    // WLI2H3 = vector<vector<vector<float>>>();
    // ALH1H2 = vector<vector<bool>>();
    // ALH1H3 = vector<vector<bool>>();
    // ALH2H3 = vector<vector<bool>>();
    // WLH1H2 = vector<vector<float>>();
    // WLH1H3 = vector<vector<float>>();
    // WLH2H3 = vector<vector<float>>();
    // ALH1OL = vector<vector<vector<bool>>>();
    // ALH2OL = vector<vector<vector<bool>>>();
    // ALH3OL = vector<vector<vector<bool>>>();
    // WLH1OL = vector<vector<vector<float>>>();
    // WLH2OL = vector<vector<vector<float>>>();
    // WLH3OL = vector<vector<vector<float>>>();

    // vector<bool>().swap(HLA1);
    // vector<bool>().swap(HLA2);
    // vector<bool>().swap(HLA3);
    // vector<vector<vector<vector<bool>>>>().swap(ALI0OL);
    // vector<vector<vector<vector<bool>>>>().swap(ALI1OL);
    // vector<vector<vector<vector<bool>>>>().swap(ALI2OL);
    // vector<vector<vector<vector<float>>>>().swap(WLI0OL);
    // vector<vector<vector<vector<float>>>>().swap(WLI1OL);
    // vector<vector<vector<vector<float>>>>().swap(WLI2OL);
    // vector<vector<vector<bool>>>().swap(ALI0H1);
    // vector<vector<vector<bool>>>().swap(ALI1H1);
    // vector<vector<vector<bool>>>().swap(ALI2H1);
    // vector<vector<vector<float>>>().swap(WLI0H1);
    // vector<vector<vector<float>>>().swap(WLI1H1);
    // vector<vector<vector<float>>>().swap(WLI2H1);
    // vector<vector<vector<bool>>>().swap(ALI0H2);
    // vector<vector<vector<bool>>>().swap(ALI1H2);
    // vector<vector<vector<bool>>>().swap(ALI2H2);
    // vector<vector<vector<float>>>().swap(WLI0H2);
    // vector<vector<vector<float>>>().swap(WLI1H2);
    // vector<vector<vector<float>>>().swap(WLI2H2);
    // vector<vector<vector<bool>>>().swap(ALI0H3);
    // vector<vector<vector<bool>>>().swap(ALI1H3);
    // vector<vector<vector<bool>>>().swap(ALI2H3);
    // vector<vector<vector<float>>>().swap(WLI0H3);
    // vector<vector<vector<float>>>().swap(WLI1H3);
    // vector<vector<vector<float>>>().swap(WLI2H3);
    // vector<vector<bool>>().swap(ALH1H2);
    // vector<vector<bool>>().swap(ALH1H3);
    // vector<vector<bool>>().swap(ALH2H3);
    // vector<vector<float>>().swap(WLH1H2);
    // vector<vector<float>>().swap(WLH1H3);
    // vector<vector<float>>().swap(WLH2H3);
    // vector<vector<vector<bool>>>().swap(ALH1OL);
    // vector<vector<vector<bool>>>().swap(ALH2OL);
    // vector<vector<vector<bool>>>().swap(ALH3OL);
    // vector<vector<vector<float>>>().swap(WLH1OL);
    // vector<vector<vector<float>>>().swap(WLH2OL);
    // vector<vector<vector<float>>>().swap(WLH3OL);
    // cout << "Size: " << WLH3OL.size() << ".\n";
    // cout << "Capacity: " << WLH3OL.capacity() << ".\n";
    // HLA1.resize(0);
    // HLA2.resize(0);
    // HLA3.resize(0);
    // ALI0OL.resize(0);
    // ALI1OL.resize(0);
    // ALI2OL.resize(0);
    // WLI0OL.resize(0);
    // WLI1OL.resize(0);
    // WLI2OL.resize(0);
    // ALI0H1.resize(0);
    // ALI2H1.resize(0);
    // WLI0H1.resize(0);
    // WLI1H1.resize(0);
    // WLI2H1.resize(0);
    // ALI0H2.resize(0);
    // ALI1H2.resize(0);
    // ALI2H2.resize(0);
    // WLI0H2.resize(0);
    // WLI1H2.resize(0);
    // WLI2H2.resize(0);
    // ALI0H3.resize(0);
    // ALI1H3.resize(0);
    // ALI2H3.resize(0);
    // WLI0H3.resize(0);
    // WLI1H3.resize(0);
    // WLI2H3.resize(0);
    // ALH1H2.resize(0);
    // ALH1H3.resize(0);
    // ALH2H3.resize(0);
    // WLH1H2.resize(0);
    // WLH1H3.resize(0);
    // WLH2H3.resize(0);
    // ALH1OL.resize(0);
    // ALH2OL.resize(0);
    // ALH3OL.resize(0);
    // WLH1OL.resize(0);
    // WLH2OL.resize(0);
    // WLH3OL.resize(0);
    // cout << "Size: " << WLH3OL.size() << ".\n";
    // cout << "Capacity: " << WLH3OL.capacity() << ".\n";
    // HLA1.shrink_to_fit();
    // HLA2.shrink_to_fit();
    // HLA3.shrink_to_fit();
    // ALI0OL.shrink_to_fit();
    // ALI1OL.shrink_to_fit();
    // ALI2OL.shrink_to_fit();
    // WLI0OL.shrink_to_fit();
    // WLI1OL.shrink_to_fit();
    // WLI2OL.shrink_to_fit();
    // ALI0H1.shrink_to_fit();
    // ALI2H1.shrink_to_fit();
    // WLI0H1.shrink_to_fit();
    // WLI1H1.shrink_to_fit();
    // WLI2H1.shrink_to_fit();
    // ALI0H2.shrink_to_fit();
    // ALI1H2.shrink_to_fit();
    // ALI2H2.shrink_to_fit();
    // WLI0H2.shrink_to_fit();
    // WLI1H2.shrink_to_fit();
    // WLI2H2.shrink_to_fit();
    // ALI0H3.shrink_to_fit();
    // ALI1H3.shrink_to_fit();
    // ALI2H3.shrink_to_fit();
    // WLI0H3.shrink_to_fit();
    // WLI1H3.shrink_to_fit();
    // WLI2H3.shrink_to_fit();
    // ALH1H2.shrink_to_fit();
    // ALH1H3.shrink_to_fit();
    // ALH2H3.shrink_to_fit();
    // WLH1H2.shrink_to_fit();
    // WLH1H3.shrink_to_fit();
    // WLH2H3.shrink_to_fit();
    // ALH1OL.shrink_to_fit();
    // ALH2OL.shrink_to_fit();
    // ALH3OL.shrink_to_fit();
    // WLH1OL.shrink_to_fit();
    // WLH2OL.shrink_to_fit();
    // WLH3OL.shrink_to_fit();
    // cout << "Size: " << WLH3OL.size() << ".\n";
    // cout << "Capacity: " << WLH3OL.capacity() << ".\n";
}
//------------------------------PopulationCVRP---------------------------------------------------

PopulationCVRP::PopulationCVRP()
{
    initializePopulation();
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

void PopulationCVRP::saveGenomesAndFreeRAM()
{
    int i, j;
    int nSpecies = population.size();
    int nInSpecies;
    for (i = 0; i < nSpecies; ++i)
    {
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            saveGenomeInstance(population[i][j]);
            population[i][j].clearGenomeRAM();
        }
    }
}

void PopulationCVRP::saveGenomes()
{
    int i, j;
    int nSpecies = population.size();
    int nInSpecies;
    for (i = 0; i < nSpecies; ++i)
    {
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            saveGenomeInstance(population[i][j]);
        }
    }
}
void PopulationCVRP::loadAllGenomes()
{
    int i, j;
    int nSpecies = population.size();
    int nInSpecies;
    for (i = 0; i < nSpecies; ++i)
    {
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            loadGenomeInstance(population[i][j]);
        }
    }
}

void PopulationCVRP::updateGenomeIDs()
{
    genomeIDs = twoDimensionVectorCreator(0, 0, 0);
    int i, j, nSpecies, nGenomes;
    nSpecies = population.size();
    for (i = 0; i < nSpecies; ++i)
    {
        nGenomes = population[i].size();
        vector<int> speciesIDs;
        for (j = 0; j < nGenomes; ++j)
        {
            speciesIDs.push_back(population[i][j].randID);
        }
        genomeIDs.push_back(speciesIDs);
    }
}

void PopulationCVRP::IDToPopulation()
{
    population.resize(0);
    int i, j;
    int nSpecies = genomeIDs.size();
    int nGenomes;
    for (i = 0; i < nSpecies; ++i)
    {
        nGenomes = genomeIDs[i].size();
        cout << "Number of genomes in species " << i << ": " << nGenomes << "\n";
        vector<GenomeCVRP> species;
        for (j = 0; j < nGenomes; ++j)
        {
            GenomeCVRP g;
            g.randID = genomeIDs[i][j];
            species.push_back(g);
        }
        population.push_back(species);
    }
}
template <class Archive>
void PopulationCVRP::serialize(Archive &ar, const unsigned int file_version)
{
    // boost::serialization::split_member(ar, *this, file_version);
    ar & genomeIDs & generation & nCostumers;
}
void savePopulation(PopulationCVRP &population)
{
    filesystem::path directoryPath = string(DIRECTORY) + "data/";
    deleteDirectoryContents(directoryPath);
    population.updateGenomeIDs();
    population.saveGenomes();

    string filename = string(DIRECTORY) + "data/Population.dat";
    std::ofstream outfile(filename);
    boost::archive::text_oarchive archive(outfile);

    archive & population;
}

void loadPopulation(PopulationCVRP &population)
{
    cout << "Loading population from : " << DIRECTORY << "\n";
    string filename = string(DIRECTORY) + "data/Population.dat";
    std::ifstream infile(filename);
    boost::archive::text_iarchive archive(infile);
    archive & population;
    cout << "Loading genomes.\n";
    population.IDToPopulation();
    population.loadAllGenomes();
}

/// @brief This method will let all the genomes run on the current CVRP istance. It starts by letting cplex run with its standard branching scheme.If it doesn branch return false.
bool PopulationCVRP::runCPLEXGenome()
{
    createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 0, defaultCPLEXPlaceholder, 60);
    cout << "Nodes CPLEX: " << defaultCPLEXPlaceholder.fitness[0] << ".\n";
    if (defaultCPLEXPlaceholder.fitness[0] == 0 || defaultCPLEXPlaceholder.fitness[0] == 1000000)
    {
        return false; // There was no branching in this instance or cplex didnt find a feasible solution
    }
#pragma omp parallel num_threads(4)
#pragma omp single
    {
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 1, minInfeasibleCPLEXPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 2, maxInfeasibleCPLEXPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 3, pseudoCPLEXPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 4, strongCPLEXPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 5, pseudoReducedCPLEXPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 6, heuristicShortestPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 7, heuristicLongestPlaceholder, 60);
#pragma omp task shared(vertexMatrix, costMatrix, capacity)
        createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 8, heuristicRandomPlaceholder, 60);
    }

    cout << "----------Running on Genomes.----------\n"
         << flush;
    int i = 0;
    int j = 0;
    int nSpecies = population.size();
    int nGenomes;

    int timeoutValue; // If cplex took many nodes to find optimality we allow the genomes to take longer due to the difference in optimization of cplex and this code
    if (defaultCPLEXPlaceholder.fitness[0] < 100)
    {
        timeoutValue = 60;
    }
    else if (defaultCPLEXPlaceholder.fitness[0] < 300)
    {
        timeoutValue = 120;
    }
    else if (defaultCPLEXPlaceholder.fitness[0] < 500)
    {
        timeoutValue = 180;
    }
    else if (defaultCPLEXPlaceholder.fitness[0] < 1000)
    {
        timeoutValue = 240;
    }
    else
    {
        timeoutValue = 300;
    }

#pragma omp parallel for private(i, nGenomes, j) shared(vertexMatrix, costMatrix, capacity, population) num_threads(4)
    for (i = 0; i < nSpecies; ++i)
    {
        nGenomes = population[i].size();
        for (j = 0; j < nGenomes; ++j)
        {
            createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 9, population[i][j], timeoutValue);
        }
    }
    return true;
}

void PopulationCVRP::initializePopulation()
{
    for (int i = 0; i < 5; ++i)
    {
        vector<GenomeCVRP> temp;
        population.push_back(temp);
    }
    int speciesNumber = 0;

    for (int i = 0; i < populationSize; ++i)
    {
        speciesNumber = i % 5;
        GenomeCVRP g;
        g.countActiveConnections();
        population[speciesNumber].push_back(g);
    }
}

void PopulationCVRP::reorganizeSpecies()
{

    // For each species there is a representative, we calculate the distance between the genome and the representative
    // If the distance is small enough we add the genome to the given species
    // We find the likeliness between each genome and each species representative. We then calculate the outlier value for the best likeliness
    // We add the genome to the species it is most close to and to a new species if it isnt like any other species

    int i, j;
    vector<GenomeCVRP> tempVec;
    bool foundSpecies;
    vector<vector<GenomeCVRP>>::iterator it;
    for (i = 0; i < population.size(); i++)
    {
        for (j = 0; j < population[i].size(); ++j)
        {
            if (j == 0)
            {
                continue;
            }
            tempVec.push_back(population[i][j]);
        }
        if (population[i].size() > 4)
        {
            population[i].resize(1); // Only the species representative remains, if the species had at least 5 genomes in it
        }
        else
        {
            tempVec.push_back(population[i][0]);
            population.erase(population.begin() + i);
            --i;
        }
    }
    vector<vector<GenomeCVRP>> tempPop = population;
    vector<vector<float>> likeness(population.size(), vector<float>(tempVec.size(), 0));
    j = 0;
    for (GenomeCVRP g : tempVec) // Each genome that is not a species representative must now be allocated
    {
        foundSpecies = false;
        for (i = 0; i < population.size(); ++i) // Compare to each species representative
        {
            likeness[i][j] = calculateLikeness(g, population[i][0]);
        }
        ++j;
    }
    vector<float> bestLikeness(likeness[i].size(), 0);
    vector<float> orderdLikeness(likeness[i].size(), 0);
    for (i = 0; i < likeness[i].size(); ++i)
    {
        for (j = 0; j < likeness.size(); ++j)
        {
            if (likeness[i][j] < bestLikeness[i])
            {
                bestLikeness[i] = likeness[i][j];
                orderdLikeness[i] = likeness[i][j];
            }
        }
    }
    sort(orderdLikeness.begin(), orderdLikeness.end());
    i = 0.25 * orderdLikeness.size();
    j = 0.75 * orderdLikeness.size();
    float outlierTreshold = orderdLikeness[j] + 1.5 * (orderdLikeness[j] - orderdLikeness[i]); // find the outlier treshold of the likeliness based on the IQR method

    vector<GenomeCVRP> vgt;
    population.push_back(vgt); // create a new species for the outcasts
    int nSpecies = population.size();
    for (i = 0; i < tempVec.size(); ++i) // for each individual in the temporary vector
    {
        if (bestLikeness[i] > outlierTreshold) // If this genome is an outlier with respect to the other species we will create a new species for it
        {
            population[(nSpecies - 1)].push_back(tempVec[i]);
            continue;
        }
        for (j = 0; j < likeness.size(); ++j) // Find wich species it belongs to
        {
            if ((likeness[i][j] - bestLikeness[i]) < EPSU)
            {
                population[j].push_back(tempVec[i]);
            }
        }
    }
    if (population[(nSpecies - 1)].size() == 0)
    { // If there were now genomes added to the outlier species we remove it
        population.pop_back();
    }

    return;
}

/// This function calculates the likeness between two individuals and returns it as a float
/// Folowing the paper by NEAT but as C1 and C2 are both 1 we do not distinguish between Excess and Disjoint connections
float calculateLikeness(GenomeCVRP &g0, GenomeCVRP &g1)
{
    int ED = 0;  // Edge and disjolint values
    int N = 0;   // Number of active connections, the highest of the two individuals
    float W = 0; // this is the weight similarity
    int EDT = 0; // same but temporaty
    int NT = 0;
    float WT = 0;
    float likeness; // final likeness value

    // the function will use the fact that for active nodes and connections, the probability that the weight is 0 is positive but 0
    tie(WT, EDT) = fourDimensionLikeness(g0.WLI0OL, g1.WLI0OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = fourDimensionLikeness(g0.WLI1OL, g1.WLI1OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = fourDimensionLikeness(g0.WLI2OL, g1.WLI2OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI0H1, g1.WLI0H1);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI1H1, g1.WLI1H1);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI2H1, g1.WLI2H1);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI0H2, g1.WLI0H2);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI1H2, g1.WLI1H2);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI2H2, g1.WLI2H2);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI0H3, g1.WLI0H3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI1H3, g1.WLI1H3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLI2H3, g1.WLI2H3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = twoDimensionLikeness(g0.WLH1H2, g1.WLH1H2);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = twoDimensionLikeness(g0.WLH1H3, g1.WLH1H3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = twoDimensionLikeness(g0.WLH2H3, g1.WLH2H3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLH1OL, g1.WLH1OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLH2OL, g1.WLH2OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = threeDimensionLikeness(g0.WLH3OL, g1.WLH3OL);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = oneDimensionLikeness(g0.BH1, g1.BH1);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = oneDimensionLikeness(g0.BH2, g1.BH2);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = oneDimensionLikeness(g0.BH3, g1.BH3);
    W += WT;
    ED += EDT;

    tie(WT, EDT) = twoDimensionLikeness(g0.BO, g1.BO);
    W += WT;
    ED += EDT;

    if (g0.activeConnections > g1.activeConnections)
    {
        N = g0.activeConnections;
    }
    else
    {
        N = g1.activeConnections;
    }

    likeness = (((C1 * ED) / N) + (C3 * W));

    return likeness;
}

GenomeCVRP PopulationCVRP::reproduce(GenomeCVRP &g0, GenomeCVRP &g1)
{
    // g0 is by definition the more fit parent, and the disjouint and excess genes always comes from it
    // the matching genes are chosen randomly, hence we first copy the more fit parent and only need to check for the mathing genes
    GenomeCVRP gkid;
    gkid.inheritParentData(g0);

    chooseRandomForMatchingGenes(g0.WLI0OL, g1.WLI0OL, gkid.WLI0OL, g0.ALI0OL, g1.ALI0OL, gkid.ALI0OL);
    chooseRandomForMatchingGenes(g0.WLI1OL, g1.WLI1OL, gkid.WLI1OL, g0.ALI1OL, g1.ALI1OL, gkid.ALI1OL);
    chooseRandomForMatchingGenes(g0.WLI2OL, g1.WLI2OL, gkid.WLI2OL, g0.ALI2OL, g1.ALI2OL, gkid.ALI2OL);

    chooseRandomForMatchingGenes(g0.WLI0H1, g1.WLI0H1, gkid.WLI0H1, g0.ALI0H1, g1.ALI0H1, gkid.ALI0H1);
    chooseRandomForMatchingGenes(g0.WLI1H1, g1.WLI1H1, gkid.WLI1H1, g0.ALI1H1, g1.ALI1H1, gkid.ALI1H1);
    chooseRandomForMatchingGenes(g0.WLI2H1, g1.WLI2H1, gkid.WLI2H1, g0.ALI2H1, g1.ALI2H1, gkid.ALI2H1);

    chooseRandomForMatchingGenes(g0.WLI0H2, g1.WLI0H2, gkid.WLI0H2, g0.ALI0H2, g1.ALI0H2, gkid.ALI0H2);
    chooseRandomForMatchingGenes(g0.WLI1H2, g1.WLI1H2, gkid.WLI1H2, g0.ALI1H2, g1.ALI1H2, gkid.ALI1H2);
    chooseRandomForMatchingGenes(g0.WLI2H2, g1.WLI2H2, gkid.WLI2H2, g0.ALI2H2, g1.ALI2H2, gkid.ALI2H2);

    chooseRandomForMatchingGenes(g0.WLI0H3, g1.WLI0H3, gkid.WLI0H3, g0.ALI0H3, g1.ALI0H3, gkid.ALI0H3);
    chooseRandomForMatchingGenes(g0.WLI1H3, g1.WLI1H3, gkid.WLI1H3, g0.ALI1H3, g1.ALI1H3, gkid.ALI1H3);
    chooseRandomForMatchingGenes(g0.WLI2H3, g1.WLI2H3, gkid.WLI2H3, g0.ALI2H3, g1.ALI2H3, gkid.ALI2H3);

    chooseRandomForMatchingGenes(g0.WLH1H2, g1.WLH1H2, gkid.WLH1H2, g0.ALH1H2, g1.ALH1H2, gkid.ALH1H2);
    chooseRandomForMatchingGenes(g0.WLH1H3, g1.WLH1H3, gkid.WLH1H3, g0.ALH1H3, g1.ALH1H3, gkid.ALH1H3);
    chooseRandomForMatchingGenes(g0.WLH2H3, g1.WLH2H3, gkid.WLH2H3, g0.ALH2H3, g1.ALH2H3, gkid.ALH2H3);

    chooseRandomForMatchingGenes(g0.WLH1OL, g1.WLH1OL, gkid.WLH1OL, g0.ALH1OL, g1.ALH1OL, gkid.ALH1OL);
    chooseRandomForMatchingGenes(g0.WLH2OL, g1.WLH2OL, gkid.WLH2OL, g0.ALH2OL, g1.ALH2OL, gkid.ALH2OL);
    chooseRandomForMatchingGenes(g0.WLH3OL, g1.WLH3OL, gkid.WLH3OL, g0.ALH3OL, g1.ALH3OL, gkid.ALH3OL);
    return gkid;
}

void PopulationCVRP::reproducePopulation()
{
    int nGenomes = 0;
    int i, j, randomInt;
    int s = 0;
    int nSpecies = population.size();
    vector<float> sumOfSpeciesFitness(nSpecies, 0);
    vector<int> ofspringPerSpecies(nSpecies, 0);
    float totalFitness = 0;
    for (i = 0; i < nSpecies; ++i) // First we get the sum of the sizes of each species, this represents the current number of genomes in the population
    {
        s = population[i].size();
        nGenomes += s;
        for (j = 0; j < s; ++j)
        { // The number of new ofspring per species is proportional to its sum of fitness
            sumOfSpeciesFitness[i] += population[i][j].getFloatFitness();
            totalFitness += population[i][j].getFloatFitness();
        }
    }

    int delta = populationSize - nGenomes; // We will add delta genomes to the population
    int sumPopFitness = 0;
    cout << "Delta: " << delta << "\n";
    for (i = 0; i < nSpecies; ++i) // As the higher the fitness the worse the genome, we need species with lowe fitness to reperoduce more
    {
        if (population[i].size() < 2)
        {
            continue;
        }
        sumPopFitness += sumOfSpeciesFitness[i];
    }
    float sumPopFitPrime = 0;
    for (i = 0; i < nSpecies; ++i)
    {
        if (population[i].size() < 2)
        {
            continue;
        }
        sumOfSpeciesFitness[i] = (float)sumPopFitness / sumOfSpeciesFitness[i];
        sumPopFitPrime += sumOfSpeciesFitness[i];
    }
    for (i = 0; i < nSpecies; ++i)
    {
        if (population[i].size() < 2)
        {
            continue;
        }
        sumOfSpeciesFitness[i] = sumOfSpeciesFitness[i] / sumPopFitPrime;
    }

    for (i = 0; i < nSpecies; ++i)
    {
        if (population[i].size() < 2)
        {
            continue; // We need at least two genomes in a species to reproduce
        }
        ofspringPerSpecies[i] = delta * (float)sumOfSpeciesFitness[i] / totalFitness;
        delta -= ofspringPerSpecies[i];
    }

    i = 0;
    while (i < delta)
    { // As we round the number of explicilty assigned ofsping down, we need some extra and we randomly allocate them
        randomInt = rand() % nSpecies;
        if (population[randomInt].size() < 2)
        {
            continue; // We need at least two genomes in a species to reproduce
        }
        ofspringPerSpecies[randomInt]++;
        ++i;
    }

    printVector(ofspringPerSpecies);
    int parent1, parent2;
    for (i = 0; i < nSpecies; ++i)
    {
        s = population[i].size();
        for (j = 0; j < ofspringPerSpecies[i]; ++j)
        {
            // We first choose the parents
            // The first is by definition the stronger parent, or they are equal
            while (true)
            {
                parent1 = rand() % s;
                parent2 = rand() % s;
                if (parent1 == parent2)
                {
                    continue;
                }
                if (!(population[i][parent1].getFloatFitness() < population[i][parent2].getFloatFitness()))
                {
                    break; // The pair satisfies our requirements and we thus proceed to reproduction between these two parents
                }
            }
            // Create the new kid and append it to the species
            population[i].push_back(reproduce(population[i][parent1], population[i][parent2]));
        }
    }
}

void PopulationCVRP::mutatePopulation()
{
    // Mutate all the genomes in the population
    int i, j, nInSpecies;
    int nSpecies = population.size();
    float randFloat;

    for (i = 0; i < nSpecies; ++i)
    {
        nInSpecies = population[i].size();

        for (j = 0; j < nInSpecies; ++j)
        {
            randFloat = (float)(rand() % 100000) / 100000;
            if (j == 0 && nInSpecies > 4)
            { // If the species has more than 5 we copy its best genome
                continue;
            }
            if (randFloat < MUTATEWEIGHTSPROB)
            {
                population[i][j].mutateGenomeWeights();
            }
            population[i][j].mutateGenomeConnectionStructure();
            population[i][j].mutateGenomeNodeStructure();
            population[i][j].removeGenomeConnections();
        }
    }
}

void PopulationCVRP::setFloatFitnessP()
{
    int i, j, speciesSize;
    float floatFitness;
    for (i = 0; i < population.size(); ++i)
    {
        speciesSize = population[i].size();
        for (j = 0; j < speciesSize; ++j)
        {
            if (population[i][j].fitness[1] == 1)
            {                                                                           // The cplex status was optimal
                floatFitness = (float)population[i][j].fitness[0] * (float)speciesSize; // we have the number of nodes visited divided by the number of Genomes in the species
                population[i][j].setFloatFitnessG(floatFitness);
                continue;
            }
            floatFitness = (float)10 * ((float)population[i][j].fitness[0] * (float)speciesSize); // we have the number of nodes visited divided by the nnumber of Genomes in the species
            population[i][j].setFloatFitnessG(floatFitness);                                      // If the solution found by the genome is not optimal it will get penalized.
        }
    }
}
/// This method orders the species based on their fitness value, the best will be the first genome i.e. the lower fitness value the  better
void PopulationCVRP::orderSpecies()
{
    int i, j, k, l, nInSpecies, tempGenomesSize;
    float randFloat, gFitness;
    bool inserted;
    vector<GenomeCVRP>::iterator it;
    int nSpecies = population.size();
    for (i = 0; i < nSpecies; ++i) // We reorder each species
    {
        int nGenomes = population[i].size();
        if (nGenomes == 0)
        {
            continue;
        }
        vector<float> fitness(nGenomes, 0);
        vector<GenomeCVRP> tempGenomes;
        tempGenomes.push_back(population[i][0]); // Add the first genome
        for (j = 1; j < nGenomes; ++j)
        {                     // For each of the remaining genomes we check where it should be inserted
            inserted = false; // If it is the worst, it will not be inserted and thus we must place it in the back
            gFitness = population[i][j].getFloatFitness();
            for (k = 0; k < tempGenomes.size(); ++k)
            {
                if (gFitness > tempGenomes[k].getFloatFitness())
                { // if lower should be further in the list
                    continue;
                }
                it = tempGenomes.begin() + k;
                tempGenomes.insert(it, population[i][j]);
                inserted = true;
                break;
            }
            if (!inserted)
            {
                tempGenomes.push_back(population[i][j]);
            }
        }
        population[i] = tempGenomes;
    }

    for (i = 0; i < nSpecies; ++i)
    {
        cout << "Species: " << i << "\n";
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            cout << population[i][j].getFloatFitness() << "\n";
        }
    }
}

void PopulationCVRP::setLogFile(string filepath)
{
    logFile = filepath;
}

// TODO Create function that reports and log what happened in the current generation (one log file is better to use later)
void PopulationCVRP::log(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers, bool branched)
{
    int i;
    ofstream ofile(logFile, ios::out | ios::app);
    ofile << "\nGeneration number: " << generation << ".\n";
    if (!branched)
    {
        ofile << "With depot location: " << depotLocation << ", customer distribution: " << customerDistribution << ", demand distribution: " << demandDistribution << " and number of costumers: " << nCostumers << ".\n";
        ofile << "No branching needed in this instance, moving on.\n";
        return;
    }

    ofile << "With depot location: " << depotLocation << ", customer distribution: " << customerDistribution << ", demand distribution: " << demandDistribution << " and number of costumers: " << nCostumers << ".\n";
    ofile << "Number of node of default branching: " << defaultCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Default branching found optimal: " << defaultCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Default branching found value: " << defaultCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of node of minfeasible branching: " << minInfeasibleCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Minfeasible branching found optimal: " << minInfeasibleCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Minfeasible branching found value: " << minInfeasibleCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of node of maxfeasible branching: " << maxInfeasibleCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Maxfeasible branching found optimal: " << maxInfeasibleCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Maxfeasible branching found value: " << maxInfeasibleCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of node of pseudo branching: " << pseudoCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Pseudo branching found optimal: " << pseudoCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Pseudo branching found value: " << pseudoCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of node of strong branching: " << strongCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Strong branching found optimal: " << strongCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Strong branching found value: " << strongCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of node of pseudoreduced branching: " << pseudoReducedCPLEXPlaceholder.fitness[0] << ".\n";
    ofile << "Pseudoreduced branching found optimal: " << pseudoReducedCPLEXPlaceholder.fitness[1] << ".\n";
    ofile << "Pseudoreduced branching found value: " << pseudoReducedCPLEXPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of nodes of shortest distance heuristic: " << heuristicShortestPlaceholder.fitness[0] << ".\n";
    ofile << "Shortest heuristic found optimal: " << heuristicShortestPlaceholder.fitness[1] << ".\n";
    ofile << "Shortest heuristic found value: " << heuristicShortestPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of nodes of longest distance heuristic: " << heuristicLongestPlaceholder.fitness[0] << ".\n";
    ofile << "Longest heuristic found optimal: " << heuristicLongestPlaceholder.fitness[1] << ".\n";
    ofile << "Longest heuristic found value: " << heuristicLongestPlaceholder.fitness[2] << ".\n\n";

    ofile << "Number of nodes of random heuristic: " << heuristicRandomPlaceholder.fitness[0] << ".\n";
    ofile << "Random heuristic found optimal: " << heuristicRandomPlaceholder.fitness[1] << ".\n";
    ofile << "Random heuristic found value: " << heuristicRandomPlaceholder.fitness[2] << ".\n\n";

    for (i = 0; i < population.size(); ++i)
    {
        ofile << "Number of nodes of best of species " << i << ": " << population[i][0].fitness[0] << ".\n";
        ofile << "Best of species " << i << " found optimal: " << population[i][0].fitness[1] << ".\n";
        ofile << "Best of species " << i << " found value: " << population[i][0].fitness[2] << ".\n";
    }
}

void PopulationCVRP::logTable(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers)
{
    string logFileTable = (string)DIRECTORY + "tables.txt";
    ofstream ofile(logFileTable, ios::out | ios::app);

    int i, j, nSpecies, nGenomes, maxGenomes;
    nSpecies = population.size();
    maxGenomes = 8; // We need at least 3, as there is at least cplex heursitc short and long
    for (i = 0; i < nSpecies; ++i)
    { // Find the species that has the most genomes
        nGenomes = population[i].size();
        if (nGenomes > maxGenomes)
        {
            maxGenomes = nGenomes;
        }
    }
    ofile << "\n\\begin{table}[ht]\n";
    ofile << "\\centering\n";
    ofile << "\\begin{tabular}{|";
    for (i = 0; i <= maxGenomes; ++i)
    {
        ofile << "l|";
    }
    ofile << "}\n ";

    ofile << "\\hline\n";

    ofile << "MISC & ";
    ofile << defaultCPLEXPlaceholder.fitness[0] << "/" << defaultCPLEXPlaceholder.fitness[2] << " & ";
    ofile << minInfeasibleCPLEXPlaceholder.fitness[0] << "/" << minInfeasibleCPLEXPlaceholder.fitness[2] << " & ";
    ofile << maxInfeasibleCPLEXPlaceholder.fitness[0] << "/" << maxInfeasibleCPLEXPlaceholder.fitness[2] << " & ";
    ofile << pseudoCPLEXPlaceholder.fitness[0] << "/" << pseudoCPLEXPlaceholder.fitness[2] << " & ";
    ofile << strongCPLEXPlaceholder.fitness[0] << "/" << strongCPLEXPlaceholder.fitness[2] << " & ";
    ofile << pseudoReducedCPLEXPlaceholder.fitness[0] << "/" << pseudoReducedCPLEXPlaceholder.fitness[2] << " & ";
    ofile << heuristicShortestPlaceholder.fitness[0] << "/" << heuristicShortestPlaceholder.fitness[2] << " & ";
    ofile << heuristicRandomPlaceholder.fitness[0] << "/" << heuristicRandomPlaceholder.fitness[2] << " & ";
    if (maxGenomes == 9)
    { // treat it as a last column
        ofile << heuristicLongestPlaceholder.fitness[0] << "/" << heuristicLongestPlaceholder.fitness[2] << " \\\\ ";
    }
    else
    {
        ofile << heuristicLongestPlaceholder.fitness[0] << "/" << heuristicLongestPlaceholder.fitness[2] << " & ";
    }

    for (i = 9; i < maxGenomes; ++i)
    {
        if (i == (maxGenomes - 1))
        { // Check if it is the last column
            ofile << "-/- \\\\ ";
        }
        else
        { // If it is not the last column
            ofile << "-/- & ";
        }
    }
    ofile << "\\hline\n";

    for (i = 0; i < nSpecies; ++i)
    {
        ofile << "Species " << i << " & ";

        nGenomes = population[i].size();
        for (j = 0; j < maxGenomes; ++j)
        {
            if (j < nGenomes)
            { // This genome exists
                if (j == (maxGenomes - 1))
                { // Check if it is the last column
                    ofile << population[i][j].fitness[0] << "/" << population[i][j].fitness[2] << " \\\\ ";
                }
                else
                { // If it is not the last column
                    ofile << population[i][j].fitness[0] << "/" << population[i][j].fitness[2] << " & ";
                }
                continue;
            }
            if (j == (maxGenomes - 1))
            { // Check if it is the last column
                ofile << "-/- \\\\ ";
            }
            else
            { // If it is not the last column
                ofile << "-/- & ";
            }
        }
        ofile << "\\hline\n";
    }
    ofile << "\\end{tabular}\n";
    ofile << "\\caption{Generation: " << generation << "}\n";
    ofile << "\\label{TableGeneration: " << generation << "}\n";
    ofile << "\\end{table}\n";

    ofile << "With depot location: " << depotLocation << ", customer distribution: " << customerDistribution << ", demand distribution: " << demandDistribution << " and number of costumers: " << nCostumers << ".\n";
}

void PopulationCVRP::logSemiColonSeparated(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers)
{

    int i, j, nSpecies, nGenomes;
    nSpecies = population.size();
    string logFileTable = (string)DIRECTORY + "logSemiColonSeparated.csv";
    ofstream ofile(logFileTable, ios::out | ios::app);
    int cplexFitness = defaultCPLEXPlaceholder.fitness[2];
    ofile << depotLocation << ";" << customerDistribution << ";" << demandDistribution << ";" << nCostumers << ";";
    ofile << defaultCPLEXPlaceholder.fitness[0] << ";";
    ofile << minInfeasibleCPLEXPlaceholder.fitness[0] << ";";
    ofile << maxInfeasibleCPLEXPlaceholder.fitness[0] << ";";
    ofile << pseudoCPLEXPlaceholder.fitness[0] << ";";
    ofile << strongCPLEXPlaceholder.fitness[0] << ";";
    ofile << pseudoReducedCPLEXPlaceholder.fitness[0] << ";";
    ofile << heuristicShortestPlaceholder.fitness[0] << ";";
    ofile << heuristicLongestPlaceholder.fitness[0] << ";";
    ofile << heuristicRandomPlaceholder.fitness[0] << ";";
    for (i = 0; i < nSpecies; ++i)
    {
        nGenomes = population[i].size();
        for (j = 0; j < nGenomes; ++j)
        {
            if (population[i][j].fitness[2] == cplexFitness)
            {
                ofile << population[i][j].fitness[0] << ";";
            }
        }
    }
    ofile << "\n";
}

void PopulationCVRP::logBestSplitOrder(string depotLocation, string customerDistribution, int demandDistribution, int nCostumers)
{
    int i, j, nSpecies, nGenomes, bestI, bestJ, Child1, Child2;
    nSpecies = population.size();
    string logFileTable = (string)DIRECTORY + "logBestSplitOrder.txt";
    ofstream ofile(logFileTable, ios::out | ios::app);
    int cplexFitness = defaultCPLEXPlaceholder.fitness[2];
    ofile << "With depot location: " << depotLocation << ", customer distribution: " << customerDistribution << ", demand distribution: " << demandDistribution << " and number of costumers: " << nCostumers << ".\n";
    int bestNnodes = 1000000;
    for (i = 0; i < nSpecies; ++i)
    {
        nGenomes = population[i].size();
        for (j = 0; j < nGenomes; ++j)
        {
            if (population[i][j].fitness[0] < bestNnodes)
            {
                bestNnodes = population[i][j].fitness[0];
                bestI = i;
                bestJ = j;
            }
        }
    }
    vector<vector<int>> resultOrdered = fromEdgeUsageToRouteSolution(population[bestI][bestJ].result);
    ofile << "Result routes:\n";
    for (vector<int> row : resultOrdered)
    {
        ofile << "D->";
        for (int val : row)
        {
            ofile << val << "->";
        }
        ofile << "D\n";
    }
    vector<int> toProcess;
    int parent = 0;
    int counter = 0;
    do
    {
        counter++;
        if (population[bestI][bestJ].searchTreeParentToChildren.count(parent) == 1)
        {
            tie(Child1, Child2, i, j) = population[bestI][bestJ].searchTreeParentToChildren[parent];
            if (population[bestI][bestJ].result[i][j] == 1)
            {
                toProcess.push_back(Child1);
                ofile << "Parent: " << parent << ", Child In Path: " << Child1 << ", Split Active: " << i << ", " << j << "\n";
            }
            else
            {
                toProcess.push_back(Child2);
                ofile << "Parent: " << parent << ", Child In Path: " << Child2 << ", Split Inactive: " << i << ", " << j << "\n";
            }
        }
        else
        {
            break;
        }
        parent = toProcess[(toProcess.size() - 1)];
    } while (counter < 100);
}

void PopulationCVRP::decimate(float percentage)
{
    vector<float> fitnesses(populationSize, 0);
    int i, j, nInSpecies;
    int k = 0;
    int nSpecies = population.size();
    for (i = 0; i < nSpecies; ++i)
    {
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            fitnesses[k] = population[i][j].getFloatFitness();
            ++k;
        }
    }
    sort(fitnesses.begin(), fitnesses.end());                       // Sorts in increasing order
    int decimationInt = (float)populationSize * (1.0 - percentage); // This will be the number of individuals not to be eliminated (total minus eliminated)
    float threshold = fitnesses[(decimationInt - 1)];               // Get the value of the worst genome not to be eleiminated

    vector<GenomeCVRP>::iterator it;
    for (i = 0; i < nSpecies; ++i) // eliminate the genomes
    {
        nInSpecies = population[i].size();
        for (j = 0; j < nInSpecies; ++j)
        {
            if (population[i][j].getFloatFitness() < threshold) // As the higher the fitness the worst the genome, eliminate only if above threshold
            {
                continue;
            }
            if (decimationInt == populationSize)
            {
                break;
            }
            it = population[i].begin() + j;
            population[i].erase(it);
            --j; // Decrease j by one if we have removed the current Genome, as the following genomes are all moved one up
            --nInSpecies;
            ++decimationInt;
        }
    }
}

void PopulationCVRP::decimatePreserveSpecies(float percentage)
{
    vector<float> fitnesses(populationSize, 0);
    int i, j, nInSpecies, worstI, worstJ;
    float worstFitness;
    int nSpecies = population.size();
    int toBeEliminated = populationSize * percentage; // This will be the number of individuals not to be eliminated (total minus eliminated)
    vector<GenomeCVRP>::iterator it;
    do
    { // Eliminate genomes untill enough are gone
        worstFitness = INT_MAX;
        for (i = 0; i < nSpecies; ++i) // eliminate the genomes
        {
            if (population[i].size() < 3)
            { // Do not eliminate from population with fewer than three genomes (every species needs at least two genomes two reproduce)
                continue;
            }

            nInSpecies = population[i].size();
            for (j = 0; j < nInSpecies; ++j)
            {
                if (population[i][j].getFloatFitness() < worstFitness) // As the higher the fitness the worst the genome, eliminate only if above threshold
                {
                    worstI = i;
                    worstJ = j;
                    worstFitness = population[i][j].getFloatFitness();
                }
            }
        }
        it = population[worstI].begin() + worstJ;
        population[worstI].erase(it);
        --toBeEliminated;
    } while (!(toBeEliminated == 0));
}

/// @brief This method will eliminate all the genomes from the smallest population and use the best ones of the remaining to create a new population
void PopulationCVRP::eliminateSpeciesAndMakeNew()
{
    int i, j, nInSpecies, smallestI, smallestPopulationNumber;
    smallestPopulationNumber = populationSize; // the smallest population is the one that has been performing the worst
    int nSpecies = population.size();

    for (i = 0; i < nSpecies; ++i) // eliminate the genomes
    {
        cout << "Species " << i << " has size: " << population[i].size() << ".\n";
        if (population[i].size() < smallestPopulationNumber)
        {
            smallestI = i;
            smallestPopulationNumber = population[i].size();
        }
    }
    cout << "Smallest population is  " << smallestI << " with size: " << smallestPopulationNumber << ".\n";

    population[smallestI].resize(0); // eliminate all the genomes from the population
    i = 0;                           // is the species we are currently looking to add
    j = 0;                           // the genome of the species (starts with the best, if all best have been added and the population is not yet complete add the second best and so forth)
    do
    { // Here we add the best genomes from the other species untill the population is back to its original size
        if (i == smallestI)
        { // Cant add from the species that has just been removed, hence continue
            ++i;
            continue;
        }
        if (!(i < population.size()))
        { // If we have checked all the species go back to first and get the second best genome
            i = 0;
            ++j;
        }
        if (!(j < population[i].size()))
        {
            continue;
        }
        cout << "Transfering genome " << j << " from species " << i << ".\n"
             << flush;
        GenomeCVRP g; // create new genome
        g.inheritParentData(population[i][j]);
        cout << "Genome transfered.\n"
             << flush;
        population[smallestI].push_back(g);

        ++i;                        // Go to next species
        --smallestPopulationNumber; // Need to add one less genome to get back to species size
    } while (!(smallestPopulationNumber == 0));
}
// TODO Create a function that runs all the steps of the Evolution
void PopulationCVRP::evolutionLoop()
{
    string depotLocation, customerDistribution;
    int demandDistribution;
    vector<tuple<string, string, int>> permutations = getCVRPPermutations();
    bool branched = true;
    int nSpecies, species, i;
    while (true) // We have an endless training loop
    {
        for (tuple<string, string, int> instanceValues : permutations) // For each of the permutations of the CVRP we will run all the steps once
        {
            // depotLocation = "R";
            // customerDistribution = "R";
            // demandDistribution = 3;
            depotLocation = get<0>(instanceValues);
            customerDistribution = get<1>(instanceValues);
            demandDistribution = get<2>(instanceValues);

            getCVRP(nCostumers, depotLocation, customerDistribution, demandDistribution);

            cout << "Generation: " << generation << "\n";
            cout << "depotLocation: " << depotLocation << "\n";
            cout << "customerDistribution: " << customerDistribution << "\n";
            cout << "demandDistribution: " << demandDistribution << "\n";
            cout << "NCustomers: " << nCostumers << "\n";
            branched = runCPLEXGenome();
            if (!branched)
            { // If it hasnt branched we take another instance
                cout << "Instance required no branching, getting new CRVP instance.\n";
                log(depotLocation, customerDistribution, demandDistribution, nCostumers, false);
                continue;
            }
            if (generation % 5 == 0)
            {
                cout << "Saving population.\n"
                     << flush;
                savePopulation(*this);
            }

            cout << "Setting float fitness.\n"
                 << flush;
            setFloatFitnessP();
            cout << "Ordering genomes.\n"
                 << flush;
            orderSpecies();
            cout << "Logging.\n"
                 << flush;
            log(depotLocation, customerDistribution, demandDistribution, nCostumers, true); // makes log clearer if in order
            logTable(depotLocation, customerDistribution, demandDistribution, nCostumers);
            logSemiColonSeparated(depotLocation, customerDistribution, demandDistribution, nCostumers);
            logBestSplitOrder(depotLocation, customerDistribution, demandDistribution, nCostumers);
            // cout << "Reorganizing genomes into species.\n"
            //      << flush;
            // reorganizeSpecies();
            cout << "Decimating.\n"
                 << flush;
            decimatePreserveSpecies(0.3);
            if (generation % 10 == 0)
            { // Every tenth generation we remove a species and create a new one made out of the best genomes of the remaining species
                cout << "Eliminating species and creating new one!\n"
                     << flush;
                eliminateSpeciesAndMakeNew();
            }
            // decimate(0.15);
            cout << "Reproducing.\n"
                 << flush;
            reproducePopulation();
            cout << "Mutating genomes.\n"
                 << flush;
            mutatePopulation();
            nSpecies = population.size();
            for (i = 0; i < nSpecies; ++i)
            {
                cout << "Population " << i << " has size " << population[i].size() << "\n";
            }

            cout << "Generation ready for next cycle.\n-------------------------------------------------------------------------\n"
                 << flush;

            ++generation;
        }
        // ++nCostumers;
        if (nCostumers == population[0][0].maxInputSize)
        { // Dont let ther be more costumers than the genome can learn on (It can run on more but doesnt take the last values into account then)
            nCostumers = 10;
        }
    }
}

//------------------------------CVRPCALLBACK---------------------------------------------------
CVRPCallback::CVRPCallback(){};
CVRPCallback::CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                           const NumVarMatrix &_edgeUsage, const vector<vector<int>> &_costMatrix, const vector<vector<int>> &_coordinates, const GenomeCVRP &_genome, const int genomeBool) : edgeUsage(_edgeUsage)
{
    Q = capacity;
    N = n;
    T = genomeBool;
    demandVector = demands;
    genome = _genome;
    M1 = twoDimensionVectorCreator(50, 50, (float)0);
    coordinates = twoDimensionVectorCreator(3, 50, (float)0);
    misc = twoDimensionVectorCreator(1, 25, (float)0);

    int ioTime = 0;
    int nnTime = 0;
    // Create the first part of M1 (i.e. the distance(cost) between each node)
    //(This is a half constant input for the genome)
    for (int i = 0; i < N; ++i)
    {
        for (int j = (i + 1); j < N; ++j)
        {
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
CVRPCallback::branchingWithCPLEX() const{

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
    currentNodeID = context.getLongInfo(IloCplex::Callback::Context::Info::NodeUID);
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
        auto start = chrono::high_resolution_clock::now();
        for (IloInt i = 1; i < N; ++i) // This will get the relaxation value for each edge, as the matrix is symmetric we only fill one side
        {
            for (IloInt j = 0; j < i; j++)
            {
                M1[i][j] = context.getRelaxationPoint(edgeUsage[i][j]);
            }
        }
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        ioTime += duration.count();
        // cout <<"TIME: "<<time<<"\n"<<flush;
        misc[0][0] = (float)context.getLongInfo(IloCplex::Callback::Context::Info::NodeDepth); // Add the depth to the first variable of miscelaneous
        // misc[0][1] = (float)abs()/;

        //------------------GENOME-----------------------
        // cout << "\n----GENOME----\n";
        // cout << "Current node: " << currentNodeID << "\n";
        CPXLONG upChild, downChild;
        double const up = 1;
        double const down = 0;
        // we make a forward pass and get the i and j values, and get the cplex variable to split
        vector<vector<bool>> splittedBefore = getEdgeBranches();
        int iSplit, jSplit;

        start = chrono::high_resolution_clock::now();
        tie(iSplit, jSplit) = genome.feedForwardFirstTry(&M1, &coordinates, &misc, &splittedBefore);
        stop = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        nnTime += duration.count();
        IloNumVar branchVar = edgeUsage[iSplit][jSplit];

        // cout << "\ni: " << iSplit << ";  j: " << jSplit << "\n"
        //      << flush;
        // cout << "____GENOME____\n"
        //      << flush;

        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        // Insert the children as key into the structure map, with the parent (current node) as value
        nodeStructure[upChild] = currentNodeID;
        nodeStructure[downChild] = currentNodeID;
        nodeStructureParentToChilds[currentNodeID] = make_tuple(upChild, downChild, iSplit, jSplit);
        // Insert the current node (as key) in the map, with the branched variabels (edge) as value
        nodeSplit[currentNodeID] = make_tuple(iSplit, jSplit);
        // cout << "KU: " << upChild << " KD: " << downChild << "\n"
        //      << flush;
        mtxCVRP.unlock();
    }
}
inline void
CVRPCallback::branchingWithShortestHeuristic(const IloCplex::Callback::Context &context) const
{
    // Get the current relaxation.
    // The function not only fetches the objective value but also makes sure
    // the node lp is solved and returns the node lp's status. That status can
    // be used to identify numerical issues or similar
    IloCplex::CplexStatus status = context.getRelaxationStatus(0);
    double obj = context.getRelaxationObjective();
    currentNodeID = context.getLongInfo(IloCplex::Callback::Context::Info::NodeUID);
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

        //------------------GENOME-----------------------
        // cout << "\n----GENOME----\n";
        // cout << "Current node: " << currentNodeID << "\n";
        CPXLONG upChild, downChild;
        double const up = 1;
        double const down = 0;
        // we make a forward pass and get the i and j values, and get the cplex variable to split
        vector<vector<bool>> splittedBefore = getEdgeBranches();
        int iSplit, jSplit;
        tie(iSplit, jSplit) = findClosestNotSplit(&splittedBefore);

        IloNumVar branchVar = edgeUsage[iSplit][jSplit];
        // cout << "\ni: " << iSplit << ";  j: " << jSplit << "\n"
        //      << flush;
        // cout << "____GENOME____\n"

        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        // Insert the children as key into the structure map, with the parent (current node) as value
        nodeStructure[upChild] = currentNodeID;
        nodeStructure[downChild] = currentNodeID;
        nodeStructureParentToChilds[currentNodeID] = make_tuple(upChild, downChild, iSplit, jSplit);

        // Insert the current node (as key) in the map, with the branched variabels (edge) as value
        nodeSplit[currentNodeID] = make_tuple(iSplit, jSplit);
        // cout << "KU: " << upChild << " KD: " << downChild << "\n"
        //      << flush;
        mtxCVRP.unlock();
    }
}
inline void
CVRPCallback::branchingWithLongestHeuristic(const IloCplex::Callback::Context &context) const
{
    // Get the current relaxation.
    // The function not only fetches the objective value but also makes sure
    // the node lp is solved and returns the node lp's status. That status can
    // be used to identify numerical issues or similar
    IloCplex::CplexStatus status = context.getRelaxationStatus(0);
    double obj = context.getRelaxationObjective();
    currentNodeID = context.getLongInfo(IloCplex::Callback::Context::Info::NodeUID);
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

        //------------------GENOME-----------------------
        // cout << "\n----GENOME----\n";
        // cout << "Current node: " << currentNodeID << "\n";
        CPXLONG upChild, downChild;
        double const up = 1;
        double const down = 0;
        // we make a forward pass and get the i and j values, and get the cplex variable to split
        vector<vector<bool>> splittedBefore = getEdgeBranches();
        int iSplit, jSplit;
        tie(iSplit, jSplit) = findFurthestNotSplit(&splittedBefore);

        IloNumVar branchVar = edgeUsage[iSplit][jSplit];
        // cout << "\ni: " << iSplit << ";  j: " << jSplit << "\n"
        //      << flush;
        // cout << "____GENOME____\n"

        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        // Insert the children as key into the structure map, with the parent (current node) as value
        nodeStructure[upChild] = currentNodeID;
        nodeStructure[downChild] = currentNodeID;
        nodeStructureParentToChilds[currentNodeID] = make_tuple(upChild, downChild, iSplit, jSplit);

        // Insert the current node (as key) in the map, with the branched variabels (edge) as value
        nodeSplit[currentNodeID] = make_tuple(iSplit, jSplit);
        // cout << "KU: " << upChild << " KD: " << downChild << "\n"
        //      << flush;
        mtxCVRP.unlock();
    }
}

inline void
CVRPCallback::branchingWithRandomHeuristic(const IloCplex::Callback::Context &context) const
{
    // Get the current relaxation.
    // The function not only fetches the objective value but also makes sure
    // the node lp is solved and returns the node lp's status. That status can
    // be used to identify numerical issues or similar
    IloCplex::CplexStatus status = context.getRelaxationStatus(0);
    double obj = context.getRelaxationObjective();
    currentNodeID = context.getLongInfo(IloCplex::Callback::Context::Info::NodeUID);
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

        //------------------GENOME-----------------------
        // cout << "\n----GENOME----\n";
        // cout << "Current node: " << currentNodeID << "\n";
        CPXLONG upChild, downChild;
        double const up = 1;
        double const down = 0;
        // we make a forward pass and get the i and j values, and get the cplex variable to split
        vector<vector<bool>> splittedBefore = getEdgeBranches();
        int iSplit, jSplit;
        tie(iSplit, jSplit) = findRandomNotSplit(&splittedBefore);

        IloNumVar branchVar = edgeUsage[iSplit][jSplit];
        // cout << "\ni: " << iSplit << ";  j: " << jSplit << "\n"
        //      << flush;
        // cout << "____GENOME____\n"

        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        // Insert the children as key into the structure map, with the parent (current node) as value
        nodeStructure[upChild] = currentNodeID;
        nodeStructure[downChild] = currentNodeID;
        nodeStructureParentToChilds[currentNodeID] = make_tuple(upChild, downChild, iSplit, jSplit);

        // Insert the current node (as key) in the map, with the branched variabels (edge) as value
        nodeSplit[currentNodeID] = make_tuple(iSplit, jSplit);
        // cout << "KU: " << upChild << " KD: " << downChild << "\n"
        //      << flush;
        mtxCVRP.unlock();
    }
}

tuple<int, int> CVRPCallback::findClosestNotSplit(vector<vector<bool>> *splittedBefore) const
{
    tuple<int, int> closestT;
    closestT = make_tuple(0, 1);
    float closestF = 10000.0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = (i + 1); j < N; ++j)
        {
            if (splittedBefore->at(i).at(j))
            {
                continue;
            }
            if (M1[i][j] < closestF)
            {
                closestF = M1[i][j];
                closestT = make_tuple(i, j);
            }
        }
    }
    return closestT;
}
tuple<int, int> CVRPCallback::findFurthestNotSplit(vector<vector<bool>> *splittedBefore) const
{
    tuple<int, int> furthestT;
    furthestT = make_tuple(0, 1);
    float furthestF = 0.0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = (i + 1); j < N; ++j)
        {
            if (splittedBefore->at(i).at(j))
            {
                continue;
            }
            if (M1[i][j] > furthestF)
            {
                furthestF = M1[i][j];
                furthestT = make_tuple(i, j);
            }
        }
    }
    return furthestT;
}
tuple<int, int> CVRPCallback::findRandomNotSplit(vector<vector<bool>> *splittedBefore) const
{
    tuple<int, int> furthestT;
    float furthestF = 0.0;
    int i, j;
    int counter = 0;
    while (counter < 100)
    {
        ++counter;
        i = randomUniformIntGenerator(N);
        j = randomUniformIntGenerator(N);
        if (splittedBefore->at(i).at(j))
        {
            continue;
        }
        return make_tuple(i, j);
    }
    return findClosestNotSplit(splittedBefore);
}

int CVRPCallback::getCalls() const
{
    return cutCalls;
}
int CVRPCallback::getBranches() const
{
    return branches;
}
int CVRPCallback::getLastNodeID() const
{
    return currentNodeID;
}
map<int, tuple<int, int, int, int>> CVRPCallback::getNodeStructureParentToChilds() const
{
    return nodeStructureParentToChilds;
}
map<int, int> CVRPCallback::getNodeStructureChildToParent() const
{
    return nodeStructure;
}

vector<vector<bool>> CVRPCallback::getEdgeBranches() const
{

    vector<vector<bool>> splittedBefore(N, vector<bool>(N, false));
    CPXLONG currentNode = currentNodeID;
    int i, j;
    tuple<int, int> tempT;
    while (!(currentNode == 0))
    {
        currentNode = nodeStructure[currentNode]; // The current node has not been splitted (by definition) thus we start with the imidiate parent
        tempT = nodeSplit[currentNode];
        i = get<0>(tempT);
        j = get<1>(tempT);
        // cout << "i: " << i << "; j:" << j << "\n";
        splittedBefore[i][j] = true;
        splittedBefore[j][i] = true; // symmetrical matrix
    }

    currentNode = currentNodeID;
    // cout << "StartNode: " << currentNode << "\n";
    while (!(currentNode == 0))
    {
        currentNode = nodeStructure[currentNode];
        // cout << "Node: " << currentNode << "\n";
    }
    return splittedBefore;
}

void CVRPCallback::printTimes()
{
    cout << "Total IO time: " << ioTime << ".\n";
    cout << "Total NN time: " << nnTime << ".\n";
    cout << "Total GenomeNN time: " << genome.nnTime << ".\n";
    cout << "Total invoke time: " << invokeTime << ".\n";
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
        mtxInvoke.lock();
        auto startTime = chrono::high_resolution_clock::now();
        if (T == 6)
        { // if T is equal to six we branch with heuristic with shortes path
            branchingWithShortestHeuristic(context);
        }
        else if (T == 7)
        { // if T is equal to seven we branch with heuristic with longest path
            branchingWithLongestHeuristic(context);
        }
        else if (T == 8)
        {
            branchingWithRandomHeuristic(context);
        }
        else if (T == 9)
        {
            branchingWithGenome(context);
        }else{
            branchingWithCPLEX();
        }
        auto stopTime = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
        invokeTime += duration.count();
        invokeTime++;
        mtxInvoke.unlock();

    }
}
CVRPCallback::~CVRPCallback()
{
}

NumVarMatrix populate(IloModel *model, NumVarMatrix edgeUsage, const vector<vector<int>> *vertexMatrix, vector<vector<int>> *costMatrix, int *capacity)
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
        edgeUsage[i][0].setBounds(0, 2);                     // Edges from the depot can be traveled twice, customer and back
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

tuple<vector<vector<int>>, int> createAndRunCPLEXInstance(const vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity, int runType, GenomeCVRP &genome, const int timeout)
{
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

    // cplex.setParam(IloCplex::Param::MIP::Strategy::HeuristicFreq, -1);
    CVRPCallback cgc;
    // Custom Cuts and Branch in one Custom Generic Callback
    CPXLONG contextMask = 0;
    contextMask |= IloCplex::Callback::Context::Id::Candidate;
    // If true we will use the custom branching scheme with the genome as the AI
    if (runType == 0)
    {
        ; // Let CPLEX use the default
    }
    else if (runType == 1)
    { // Use cplex min infeasible
        cplex.setParam(IloCplex::Param::MIP::Strategy::VariableSelect, -1);
    }
    else if (runType == 2)
    { // Use cplex max infeasible
        cplex.setParam(IloCplex::Param::MIP::Strategy::VariableSelect, 1);
    }
    else if (runType == 3)
    { // Use cplex pseudo cost branching
        cplex.setParam(IloCplex::Param::MIP::Strategy::VariableSelect, 2);
    }
    else if (runType == 4)
    { // Use cplex strong branching
        cplex.setParam(IloCplex::Param::MIP::Strategy::VariableSelect, 3);
    }
    else if (runType == 5)
    { // Use cplex pseudo reduced cost branching
        cplex.setParam(IloCplex::Param::MIP::Strategy::VariableSelect, 4);
    }
    else if (runType == 6 || runType == 7 || runType == 8 || runType == 9) // If its not the standard cplex we branch using custom callback
    {
        contextMask |= IloCplex::Callback::Context::Id::Branching;
    }
    cgc = CVRPCallback(capacity, n, demands, edgeUsage, costMatrix, vertexMatrix, genome, runType);

    contextMask |= IloCplex::Callback::Context::Id::Branching;
    cplex.use(&cgc, contextMask);

    cplex.setParam(IloCplex::Param::TimeLimit, 30);
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::RandomSeed, 42);

    // These are the CUTS that are standard in CPLEX,
    // we can remove them to force CPLEX to use only the custom Cut
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
    cplex.setWarning(env.getNullStream());

    int cost, nodes;
    vector<int> genomeFitness(3, 0);
    vector<vector<int>> edgeUsageSolution(n, vector<int>(n, 0));

    // Optimize the problem and obtain solution.
    if (!cplex.solve())
    {
        env.error() << "Failed to optimize LP" << endl;
        // If there is no feasible solution return high (bad) fitness values
        genomeFitness[0] = 1000000;
        genomeFitness[1] = 0;
        genomeFitness[2] = 1000000;
        genome.setFitness(genomeFitness);
        return make_tuple(edgeUsageSolution, cost);
    }

    NumVarMatrix vals(env);
    // env.out() << "Solution status = " << cplex.getStatus() << endl;
    // env.out() << "Solution value  = " << cplex.getObjValue() << endl;
    int value;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {

            value = cplex.getValue(edgeUsage[i][j]) + 0.5; // avoids the float error i.e. float 0.999... -> int 0
            edgeUsageSolution.at(i).at(j) = value;
        }
    }

    cost = cplex.getObjValue();
    nodes = cplex.getNnodes();
    cout << "Number of nodes: " << setfill(' ') << setw(5) << nodes;
    cout << ", with value: " << setfill(' ') << setw(7) << cost;
    cout << ",for runType: " << runType << "\n"
         << flush;
    genomeFitness[0] = nodes;
    if (IloAlgorithm::Status::Optimal == cplex.getStatus())
    {
        genomeFitness[1] = 1;
    }
    else
    {
        genomeFitness[1] = 0;
    }
    genomeFitness[2] = cost;
    genome.setFitness(genomeFitness);

    genome.searchTreeParentToChildren = cgc.getNodeStructureParentToChilds();
    genome.searchTreeChildToParent = cgc.getNodeStructureChildToParent();
    genome.lastID = cgc.getLastNodeID();
    genome.result = edgeUsageSolution;
    cgc.printTimes();
    env.end();
    return make_tuple(edgeUsageSolution, cost);
}

int main()
{
    srand((unsigned int)time(NULL));
    // srand(42);

    // int s = 4;                               // This generates a random number between 3 and 8
    //     vector<vector<int>> seedCoordinates(s, vector<int>(2)); // a vector containing the cluster seeds

    //     vector<vector<double>> probCoordinates(50, vector<double>(50)); // probability distributiion for customers
    //     seedCoordinates[0][0] = 10;
    //     seedCoordinates[0][1] = 10;
    //     seedCoordinates[1][0] = 22;
    //     seedCoordinates[1][1] = 10;
    //     seedCoordinates[2][0] = 10;
    //     seedCoordinates[2][1] = 40;
    //     seedCoordinates[3][0] = 30;
    //     seedCoordinates[3][1] = 30;

    //     double probSum = 0;
    //     double maxmaxmax = 0;
    //     for (int i = 0; i < 50; i++) // calculate the probability of each point as in uchoa et al.
    //     {
    //         for (int j = 0; j < 50; j++)
    //         {
    //             double probPoint = 0;
    //             for (vector<int> seed : seedCoordinates)
    //             {
    //                 if (seed[0] == i && seed[1] == j)
    //                 {                  // as all points are distinct the p of the seed location is 0
    //                     probPoint = 0; // set the p of the current point to 0
    //                     break;
    //                 }
    //                 double d = sqrt(pow((seed[0] - i), 2) + pow((seed[1] - j), 2));
    //                 double p = exp(-d / 10);
    //                 probPoint += p;
    //             }
    //             probCoordinates[i][j] = probPoint;
    //             if(probPoint>maxmaxmax){
    //                 maxmaxmax = probPoint;
    //             }
    //         }
    //     }
    //     cout << maxmaxmax << "MAXXAX\n";
    //     // double2dVectorDivisor(probCoordinates, probCoordinates[999][999]);
    // double printFloat;
    // for (vector<double> i : probCoordinates)
    // {
    //     for (double j : i)
    //     {   
    //         printFloat = j/maxmaxmax;
    //         cout << fixed << setprecision(3) << printFloat << ", ";
    //     }
    //     cout << endl;
    // }
    // exit(0);







    // GenomeCVRP g0;
    // CVRPGrapher grapher;
    // tuple<vector<vector<int>>, int> instance = generateCVRPInstance(15, "C", "CR", 3); // generate an instance of the problem
    // vector<vector<int>> vertexMatrix = get<0>(instance);
    // // vertexMatrix[0][0] = 8;
    // // vertexMatrix[0][1] = 16;
    // int capacity = get<1>(instance);
    // vector<vector<int>> costMatrix = calculateEdgeCost(&vertexMatrix);
    // vector<vector<int>> result;
    // int resultValue;
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 9, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 0, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 1, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 2, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 3, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 4, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 5, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 6, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 6, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 7, g0, 60);
    // cout << "-------------------------\n";
    // tie(result, resultValue) = createAndRunCPLEXInstance(vertexMatrix, costMatrix, capacity, 8, g0, 60);
    // vector<vector<int>> resultOrdered = fromEdgeUsageToRouteSolution(result);
    // vector<vector<int>> resultOrdered =  vector<vector<int>>(2,vector<int>(6,0));
    // resultOrdered[0][0] = 1;
    // resultOrdered[0][1] = 2;
    // resultOrdered[0][2] = 3;
    // resultOrdered[0][3] = 1;
    // resultOrdered[0].resize(4);
    // resultOrdered[1][0] = 0;
    // resultOrdered[1][1] = 4;
    // resultOrdered[1][2] = 5;
    // resultOrdered[1][3] = 6;
    // resultOrdered[1][4] = 7;
    // resultOrdered[1][5] = 0;
    // cout << "Total GNN time: " << g0.nnTime << ".\n";

    // // printVector(resultOrdered);
    // exit(0);
    // printVector(resultOrdered);
    // int current = g0.lastID;
    // int parent;
    // int Child1, Child2, i, j;
    // cout << "Last ID: " << current << "\n";
    // while(!(current==0)){
    //     //get parent
    //     parent = g0.searchTreeChildToParent[current];
    //     //print splitted
    //     tie(Child1, Child2, i, j) = g0.searchTreeParentToChildren[parent];
    //     cout << "Child1: " <<Child1<<", Child2: " <<Child2<< "\n";
    //     cout << "i: " << i << ", j:" << j << "\n";
    //     current = parent;
    // }
    // vector<int> toProcess;
    // parent = 0;
    // int counter = 0;
    // do{
    //     counter++;
    //     if (g0.searchTreeParentToChildren.count(parent) == 1){
    //         tie(Child1, Child2, i, j) = g0.searchTreeParentToChildren[parent];
    //         toProcess.push_back(Child1);
    //         toProcess.push_back(Child2);
    //         cout << "Parent: " << parent << ", Child1: " <<Child1<<", Child2: " << Child2 <<", i: " << i << ", j:" << j << "\n";
    //     }

    //     parent = toProcess[(toProcess.size()-1)];
    //     toProcess.pop_back();
    // }while(!(toProcess.size()==0) && counter<100);
    // parent = 0;
    // counter = 0;
    // do{
    //     counter++;
    //     if (g0.searchTreeParentToChildren.count(parent) == 1){
    //         // cout <<"IJ: " <<i <<"< "<<j<<"\n";
    //         tie(Child1, Child2, i, j) = g0.searchTreeParentToChildren[parent];
    //         if(result[i][j] == 1){
    //             toProcess.push_back(Child1);
    //             cout << "Parent: " << parent << ", Child In Path: " <<Child1 <<", Split Active: " <<  i << ", " << j << "\n";
    //         }else{
    //             toProcess.push_back(Child2);
    //             cout << "Parent: " << parent << ", Child In Path: " <<Child2 <<", Split Inactive: " <<  i << ", " << j << "\n";
    //         }
    //     }else{
    //         break;
    //     }
    //     parent = toProcess[(toProcess.size()-1)];
    // }while(counter<100);
    // grapher.setInstanceCoordinates(vertexMatrix);
    // // sleep(10);
    // // grapher.replot();
    // // sleep(10);
    // // grapher.setInstanceCost(resultValue);
    // grapher.setSolutionVector(resultOrdered);
    // exit(0);

    PopulationCVRP p0;
    // loadPopulation(p0);

    p0.evolutionLoop();

    return 0;
}