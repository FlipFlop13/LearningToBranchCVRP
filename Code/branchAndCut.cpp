#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/cplex.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/ilocplex.h>
#include <vector>
#include "./utilities.h"
#include "./CVRPGrapher.h"

using namespace std;

typedef IloArray<IloNumVarArray> NumVarMatrix;

#define EPS 1e-6 // epsilon useed for violation of cuts

// This is the class implementing the generic callback interface.


//This implements a custom Cut for CPLEX
//It impolements the Rounded Caapaciteted Inequalities for candidate(integer) solutions
//It is guaranteed to find a mistake if the solution is not feasible
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

    // tracks how often we have made a cut here
    IloInt cutCalls = 0;

    // Constructor with data.
    CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                 const NumVarMatrix &_edgeUsage) : edgeUsage(_edgeUsage)
    {
        Q = capacity;
        N = n;
        demandVector = demands;
    }

    // This wil check the connected components, guarantueed to find an error (if one exists)
    // for integer solutions (hence it is always called with a candidate mask)
    inline void
    connectedComponents(const IloCplex::Callback::Context &context) const
    {   
        cutCalls++;
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
            vector<int> connectedSet{i};
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

            float kks = (float)totalSetDemand / (float)Q;
            int ks = ceil(kks);
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
            kks = totalReciprocalDemand / Q;
            ks = ceil(kks);
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

    int getCalls() const {return cutCalls;}


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
        connectedComponents(context);
}

// Destructor
CVRPCallback::~CVRPCallback()
{
    cout << "BOOOOOM" << endl;
}

//***---***---***---***---***---***---***---***---***---***---***---***---***---***---***///
class BranchCallback : public IloCplex::Callback::Function
{
    IloNumVarArray x;
    int calls;
    int branches;

public:
    BranchCallback(IloNumVarArray _x) : x(_x), calls(0), branches(0)
    {
    }

    void invoke(IloCplex::Callback::Context const &context) ILO_OVERRIDE
    {
        // NOTE: Strictly speaking, the increment of calls and branches
        //       should be protected by a lock/mutex/semaphore. However, to keep
        //       the code simple we don't do that here.
        ++calls;

        // For sake of illustration prune every node that has a depth larger
        // than 1000.
        IloInt depth = context.getLongInfo(IloCplex::Callback::Context::Info::NodeDepth);
        if (depth > 1000)
        {
            context.pruneCurrentNode();
            return;
        }

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

        IloNumArray v(context.getEnv());

        // Node lp was solved to optimality. Grab the current relaxation
        // and find the most fractional variable
        context.getRelaxationPoint(x, v);
        IloInt maxVar = -1;
        IloNum maxFrac = 0.0;
        for (IloInt i = 0; i < x.getSize(); ++i)
        {
            if (x[i].getType() != IloNumVar::Float)
            {
                double const intval = ::round(v[i]);
                double const frac = ::fabs(intval - v[i]);

                if (frac > maxFrac)
                {
                    maxFrac = frac;
                    maxVar = i;
                }
            }
        }

        // If the maximum fractionality of all integer variables is small then
        // don't create a custom branch. Instead let CPLEX decide how to
        // branch.
        IloNum minFrac = 0.1;
        if (maxFrac > minFrac)
        {
            // There is a variable with a sufficiently fractional value.
            // Branch on that variable.
            CPXLONG upChild, downChild;
            double const up = ::ceil(v[maxVar]);
            double const down = ::floor(v[maxVar]);
            IloNumVar branchVar = x[maxVar];

            // Create UP branch (branchVar >= up)
            upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
            ++branches;

            // Create DOWN branch (branchVar <= down)
            downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
            ++branches;

            // We don't use the unique ids of the down and up child. We only
            // have them so illustrate how they are returned from
            // CPXcallbackmakebranch().
            (void)downChild;
            (void)upChild;
        }
        v.end();
    }

    int getCalls() const { return calls; }
    int getBranches() const { return branches; }
};

NumVarMatrix populate(IloModel *model, NumVarMatrix edgeUsage, vector<vector<int>> *vertexMatrix, vector<vector<int>> *costMatrix, int *capacity)
{
    int n = vertexMatrix->size();
    int sumOfDemands = 0;
    for (int i = 1; i < n; i++) // Get the demands for each customer
    {
        sumOfDemands += vertexMatrix->at(i).at(2);
    }
    double kkMin = sumOfDemands / *capacity;
    int kMin = ceil(kkMin); // At least 1, extra if the demand is larger than the caacity
    cout << endl
         << "Number of customers: " << n << endl;
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
    cout << "Creating CPLEX instance!" << endl;
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

    // Custom branching call
    // BranchCallback cb(x);
    // cplex.use(&cb, IloCplex::Callback::Context::Id::Branching);

    // Custom Cuts Call
    CVRPCallback cbCut(capacity, n, demands, edgeUsage);
    cplex.use(&cbCut, IloCplex::Callback::Context::Id::Candidate);

    // Tweak some CPLEX parameters so that CPLEX has a harder time to
    // solve the model and our cut separators can actually kick in.
    // cplex.setParam(IloCplex::Param::MIP::Strategy::HeuristicFreq, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::MIRCut, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::Implied, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::Gomory, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::FlowCovers, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::PathCut, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::LiftProj, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::ZeroHalfCut, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::Cliques, -1);
    // cplex.setParam(IloCplex::Param::MIP::Cuts::Covers, -1);

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

    // cplex.getValues(vals, edgeUsage);
    // env.out() << "Values        = " << vals << endl;
    // cplex.getSlacks(vals, c);
    // env.out() << "Slacks        = " << vals << endl;
    // cplex.getDuals(vals, c);
    // env.out() << "Duals         = " << vals << endl;

    // cout << "Callback was invoked " << cb.getCalls() << " times and created " << cb.getBranches() << " branches" << endl;

    // cplex.getReducedCosts(vals, x);
    // env.out() << "Reduced Costs = " << vals << endl;
    env.end();
    return make_tuple(edgeUsageSolution, cost);
}

streambuf *logging(string logging)
{
    if (logging == "yes")
    {
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
        return coutbuf;
    }
    streambuf *coutbuf = std::cout.rdbuf();
    return coutbuf;
}

int main()
{
    std::map map = readJson();
    cout << "---------------------------------------------------------\n";
    std::map<std::string, string>::iterator it = map.begin();

    // Iterate through the map and print the elements
    while (it != map.end())
    {
        std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
        ++it;
    }

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
    int nCostumers = 18;
    tuple<vector<vector<int>>, int> instance = generateCVRPInstance(nCostumers, "R", "CR", 5); // generate an instance of the problem
    // tuple<vector<vector<int>>, int> instance = readCVRP("./Graphs/X/X-n101-k25.vrp"); // generate an instance of the problem
    vector<vector<int>> customers = get<0>(instance);
    int capacity = get<1>(instance);
    vector<vector<int>> costVector = calculateEdgeCost(&customers);

    grapher.setInstanceCoordinates(customers); // Add points to the graph for each customer

    cout << "Starting..." << endl;
    vector<vector<int>> edgeUsage;
    int cost;
    tie(edgeUsage, cost) = createAndRunCPLEXInstance(customers, costVector, capacity);

    cout << "Cost: " << cost << endl;

    vector<vector<int>> solution = fromEdgeUsageToRouteSolution(edgeUsage);
    grapher.setSolutionVector(solution);
    grapher.setInstanceCost(cost);

    vector<int> demands((nCostumers + 1), 0);
    for (int i = 1; i < (nCostumers + 1); i++) // Get the demands for each customer
    {
        demands[i] = customers[i][2];
    }
    printRouteAndDemand(solution, demands);
    // sleep(100);
    return 0;
}
