#ifndef BAC_G // include guard
#define BAC_G

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/cplex.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/ilocplex.h>
#include <vector>
#include <map>
#include <mutex>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <sys/stat.h>

#include "./utilities.h"

using namespace std;

typedef IloArray<IloNumVarArray> NumVarMatrix;
typedef IloArray<IloNumArray> NumMatrix;

#define EPS 1e-6 // epsilon useed for violation of cuts

class NodeData
{
    friend class boost::serialization::access;
    int nodeUID;
    int childUp;
    int childDown;
    int branchedVariableI;
    int branchedVariableJ;
    int nodeDepth;
    double relaxationObjective;

    template <class Archive>
    void serialize(Archive &a, const unsigned version);

public:
    // // Gap?
    // double fractionOfFractionalVariables;
    // vector<vector<double>> relaxationEdgeUsage;
    NodeData(int a, int b, int c, int d, int e, int f, double g);
    int getUID();
};

// This is the class implementing the generic callback interface.

// This implements a custom Cut for CPLEX
// It impolements the Rounded Caapaciteted Inequalities for candidate(integer) solutions
// It is guaranteed to find a mistake if the solution is not feasible
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
    mutable map<int, vector<int>> tree;

public:
    // Constructor with data.
    CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                 const NumVarMatrix &_edgeUsage, bool _training);

    // This wil check the connected components, guarantueed to find an error (if one exists)
    // for integer solutions (hence it is always called with a candidate mask)
    inline void
    connectedComponents(const IloCplex::Callback::Context &context) const;

    inline void
    branching1(const IloCplex::Callback::Context &context) const;

    inline void
    branching2(const IloCplex::Callback::Context &context) const;

    void saveNodeInstance(NodeData data) const;

    int getCalls() const;
    int getBranches() const;
    map<int, vector<int>> getTree() const;

    // This is the function that we have to implement and that CPLEX will call
    // during the solution process at the places that we asked for.
    virtual void invoke(const IloCplex::Callback::Context &context) ILO_OVERRIDE;

    /// Destructor
    virtual ~CVRPCallback();
};

NumVarMatrix populate(IloModel *model, NumVarMatrix edgeUsage, vector<vector<int>> *vertexMatrix, vector<vector<int>> *costMatrix, int *capacity);

#endif /* BAC_G */