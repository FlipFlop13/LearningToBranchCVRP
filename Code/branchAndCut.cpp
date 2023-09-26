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
#include "./CVRPGrapher.h"

using namespace std;
std::mutex mtx; // mutex for critical section

typedef IloArray<IloNumVarArray> NumVarMatrix;

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
    void serialize(Archive &a, const unsigned version)
    {
        a & nodeUID & childUp & childDown & branchedVariableI & branchedVariableJ & nodeDepth & relaxationObjective;
    }

public:
    // // Gap?
    // double fractionOfFractionalVariables;
    // vector<vector<double>> relaxationEdgeUsage;
    NodeData(int a, int b, int c, int d, int e, int f, double g)
    {
        nodeUID = a;
        childUp = b;
        childDown = c;
        branchedVariableI = d;
        branchedVariableJ = e;
        nodeDepth = f;
        relaxationObjective = g;
    }
    int getUID() { return nodeUID; }
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
    mutable bool training = true;

    // tracks how often we have made a cut here
    IloInt cutCalls = 0;
    mutable IloInt branches = 0;
    mutable map<int, vector<int>> tree;

public:
    // Constructor with data.
    CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                 const NumVarMatrix &_edgeUsage, string filepath) : edgeUsage(_edgeUsage)
    {
        Q = capacity;
        N = n;
        demandVector = demands;

        filepathCGC = filepath;
        struct stat sb;
        if (!(stat(filepathCGC.c_str(), &sb) == 0)){
            mkdir(filepathCGC.c_str(), 0777);
        }
            
    }

    // This wil check the connected components, guarantueed to find an error (if one exists)
    // for integer solutions (hence it is always called with a candidate mask)
    inline void
    connectedComponents(const IloCplex::Callback::Context &context) const
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
    branching1(const IloCplex::Callback::Context &context) const
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

        // Node lp was solved to optimality. Grab the current relaxation
        // and find the most fractional variable
        // context.getRelaxationPoint(edgeUsage, v);
        IloInt maxVarI = -1;
        IloInt maxVarJ = -1;
        IloNum maxFrac = 0.0;
        IloInt n = edgeUsage.getSize();
        for (IloInt i = 0; i < n; ++i)
        {
            for (IloInt j = 0; j < n; j++)
            {
                IloNum const s = context.getRelaxationPoint(edgeUsage[i][j]);

                double const intval = ::round(s);
                double const frac = ::fabs(intval - s);

                if (frac > maxFrac)
                {
                    // cout << "Frac: " << frac << endl;
                    maxFrac = frac;
                    maxVarI = i;
                    maxVarJ = j;
                }
            }
        }

        // If the maximum fractionality of all integer variables is small then
        // don't create a custom branch. Instead let CPLEX decide how to
        // branch.
        // There is a variable with a sufficiently fractional value.
        // Branch on that variable.
        CPXLONG upChild, downChild;
        double const up = ::ceil(maxFrac);
        double const down = ::floor(maxFrac);
        IloNumVar branchVar = edgeUsage[maxVarI][maxVarJ];

        // Create UP branch (branchVar >= up)
        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);

        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);

        (void)downChild;
        (void)upChild;
        IloInt idNode = context.getLongInfo(IloCplex::Callback::Context::Info::NodeUID);
        {
            mtx.lock();
            branches++;
            branches++;
            vector<int> v = {int(upChild), int(downChild)};
            tree[idNode] = v;
            if (training)
            {
                IloInt depth = context.getLongInfo(IloCplex::Callback::Context::Info::NodeDepth);
                NodeData nd(idNode, upChild, downChild, maxVarI, maxVarJ, depth, obj);
                saveNodeInstance(nd);
            }
            mtx.unlock();
        }
    }

    inline void
    branching2(const IloCplex::Callback::Context &context) const
    {

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

        IloInt i = rand() % N;
        IloInt j = rand() % N;

        CPXLONG upChild, downChild;
        double const up = ::ceil(1);
        double const down = ::floor(0);
        IloNumVar branchVar = edgeUsage[i][j];

        // Create UP branch (branchVar >= up)
        upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
        branches++;

        // Create DOWN branch (branchVar <= down)
        downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
        branches++;
    }

    void saveNodeInstance(NodeData data) const
    {
        {
            string filename = filepathCGC + "/Node" + to_string(data.getUID()) + ".dat";
            std::ofstream outfile(filename);
            boost::archive::text_oarchive archive(outfile);
            archive << data;
        }
    }

    int getCalls() const { return cutCalls; }
    int getBranches() const { return branches; }
    map<int, vector<int>> getTree() const { return tree; }

    // This is the function that we have to implement and that CPLEX will call
    // during the solution process at the places that we asked for.
    virtual void invoke(const IloCplex::Callback::Context &context) ILO_OVERRIDE;

    /// Destructor
    virtual ~CVRPCallback();
};

// Implementation of the invoke method.
//
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
        branching1(context);
    }
}

// Destructor
CVRPCallback::~CVRPCallback()
{
    string treeFilepath = filepathCGC + "/tree.dat";
    std::ofstream ofs(treeFilepath);
    boost::archive::text_oarchive oa(ofs);
    oa << tree;
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
    cout << endl
         << "Number of customers: " << (n - 1) << endl;
    cout << "Sum of all demands: " << sumOfDemands << endl;
    cout << "Capacity: " << *capacity << endl;
    cout << "KMin: " << kMin << endl
         << endl; // The lower bound for the number of vehicles needed

    IloEnv env = model->getEnv(); // get the environment
    IloObjective obj = IloMinimize(env);

    // Create all the edge variables
    IloNumVarArray EedgeUsage = IloNumVarArray(env, n, 0, 2, ILOINT); // Edges to the depot may be travelled twice, if the route is one costumer
    edgeUsage[0] = EedgeUsage;                                        // Edges to the depot may be travelled twice, if the route is one costumer
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

tuple<vector<vector<int>>, int> createAndRunCPLEXInstance(vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity)
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
    cplex.setParam(IloCplex::Param::TimeLimit, 600);


    // Custom Cuts and Branch in one Custom Generic Callback
    CPXLONG contextMask = 0;
    contextMask |= IloCplex::Callback::Context::Id::Candidate;
    contextMask |= IloCplex::Callback::Context::Id::Branching;
    CVRPCallback cgc(capacity, n, demands, edgeUsage, "./bin/training/Instance0" );
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
    std::map jsonMap = readJson();
    std::map<std::string, string>::iterator it = jsonMap.begin();
    for (auto i : jsonMap)
    {
        cout << i.first << " " << i.second
             << endl;
    }
    // cout << jsonMap.find("customerDistribution").second() << endl;
    // cout << jsonMap.find("customerDisstribution")->second << endl;

    // This first block prepares the log file (name bassed on the current starting time)
    const std::time_t now = std::time(nullptr);
    const std::tm calendar_time = *std::localtime(std::addressof(now));
    string month = to_string(calendar_time.tm_mon + 1);
    string day = to_string(calendar_time.tm_mday);
    string hour = to_string(calendar_time.tm_hour);
    string minute = to_string(calendar_time.tm_min);
    string filename = "./log/logFile_M" + minute + "_H" + hour + "_D" + day + "_M" + month + ".txt";
    string graphname = "./Graphs/images/graph_M" + minute + "_H" + hour + "_D" + day + "_M" + month + ".png";
    ofstream out(filename);
    streambuf *coutbuf = out.rdbuf();
    streambuf *coutBufBack = std::cout.rdbuf();
    cout.rdbuf(coutbuf);
    cout << "Master Thesis Project Philip Salomons i6154933 \n";
    cout << "Learning to Branch on the Capacitated Vehicle Routing Problem \n\n";
    cout << "**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**\n";
    cout << "**--**--**--**--**--**--       LOG File           ---**--**--**--**--**--**\n\n";

    srand((unsigned int)time(NULL)); // Set the random seed as the current time, to avoid repetitions

    CVRPGrapher grapher; // create a grapher object
    int nCostumers, demandDistribution;
    string depotLocation, customerDistribution;
    tuple<vector<vector<int>>, int> instance;
    if (jsonMap["readCVRP"] == "no")
    {
        nCostumers = stoi(jsonMap["nCustomer"]);
        customerDistribution = jsonMap["customerDistribution"];
        demandDistribution = stoi(jsonMap["demandDistribution"]);
        depotLocation = jsonMap["depotLocation"];
        instance = generateCVRPInstance(nCostumers, depotLocation, customerDistribution, demandDistribution); // generate an instance of the problem
    }
    else
    {
        // string CVRPInstanceFilename = "./Graphs/X/X-n101-k25.vrp";
        string CVRPInstanceFilename = jsonMap["readCVRP"];
        instance = readCVRP(CVRPInstanceFilename); // generate an instance of the problem
    }
    vector<vector<int>> customers = get<0>(instance);
    int capacity = get<1>(instance);
    vector<vector<int>> costVector = calculateEdgeCost(&customers);

    grapher.setInstanceCoordinates(customers); // Add points to the graph for each customer

    cout << "Creating CPLEX instance..." << endl;
    vector<vector<int>> edgeUsage;
    int cost;
    tie(edgeUsage, cost) = createAndRunCPLEXInstance(customers, costVector, capacity);

    cout << "Cost: " << cost << endl;

    vector<vector<int>> solution = fromEdgeUsageToRouteSolution(edgeUsage);
    cout << "CCC";
    grapher.setSolutionVector(solution);
    grapher.setInstanceCost(cost);

    vector<int> demands((nCostumers + 1), 0);
    for (int i = 1; i < (nCostumers + 1); i++) // Get the demands for each customer
    {
        demands[i] = customers[i][2];
    }
    printRouteAndDemand(solution, demands);

    string filenameSolution = "./Graphs/LearningToLearn/LCVRP-N-15-k-6.sol";
    writeSolution(solution, cost, filenameSolution);

    writeCVRP(customers, capacity, "./Graphs/LearningToLearn/LCVRP-N-15-k-6.vrp", "LCVRP-N-15-k-6.vrp");
    return 0;
}
