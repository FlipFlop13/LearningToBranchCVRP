#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <typeinfo>
//#include "./utilities.h"
#include "./CVRPGrapher.cpp"
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/cplex.h>
#include </opt/ibm/ILOG/CPLEX_Studio_Community2211/cplex/include/ilcplex/ilocplex.h>
using namespace std;

typedef IloArray<IloNumVarArray> NumVarMatrix;

// class BranchCallback : public IloCplex::Callback::Function
// {
//     IloNumVarArray x;
//     int calls;
//     int branches;
//
// public:
//     BranchCallback(IloNumVarArray _x) : x(_x), calls(0), branches(0)
//     {
//     }
//
//     void invoke(IloCplex::Callback::Context const &context) ILO_OVERRIDE
//     {
//         // NOTE: Strictly speaking, the increment of calls and branches
//         //       should be protected by a lock/mutex/semaphore. However, to keep
//         //       the code simple we don't do that here.
//         ++calls;
//
//         // For sake of illustration prune every node that has a depth larger
//         // than 1000.
//         IloInt depth = context.getLongInfo(IloCplex::Callback::Context::Info::NodeDepth);
//         if (depth > 1000)
//         {
//             context.pruneCurrentNode();
//             return;
//         }
//
//         // Get the current relaxation.
//         // The function not only fetches the objective value but also makes sure
//         // the node lp is solved and returns the node lp's status. That status can
//         // be used to identify numerical issues or similar
//         IloCplex::CplexStatus status = context.getRelaxationStatus(0);
//         double obj = context.getRelaxationObjective();
//
//         // Only branch if the current node relaxation could be solved to
//         // optimality.
//         // If there was any sort of trouble then don't do anything and thus let
//         // CPLEX decide how to cope with that.
//         if (status != IloCplex::Optimal &&
//             status != IloCplex::OptimalInfeas)
//         {
//             return;
//         }
//
//         IloNumArray v(context.getEnv());
//
//         // Node lp was solved to optimality. Grab the current relaxation
//         // and find the most fractional variable
//         context.getRelaxationPoint(x, v);
//         IloInt maxVar = -1;
//         IloNum maxFrac = 0.0;
//         for (IloInt i = 0; i < x.getSize(); ++i)
//         {
//             if (x[i].getType() != IloNumVar::Float)
//             {
//                 double const intval = ::round(v[i]);
//                 double const frac = ::fabs(intval - v[i]);
//
//                 if (frac > maxFrac)
//                 {
//                     maxFrac = frac;
//                     maxVar = i;
//                 }
//             }
//         }
//
//         // If the maximum fractionality of all integer variables is small then
//         // don't create a custom branch. Instead let CPLEX decide how to
//         // branch.
//         IloNum minFrac = 0.1;
//         if (maxFrac > minFrac)
//         {
//             // There is a variable with a sufficiently fractional value.
//             // Branch on that variable.
//             CPXLONG upChild, downChild;
//             double const up = ::ceil(v[maxVar]);
//             double const down = ::floor(v[maxVar]);
//             IloNumVar branchVar = x[maxVar];
//
//             // Create UP branch (branchVar >= up)
//             upChild = context.makeBranch(branchVar, up, IloCplex::BranchUp, obj);
//             ++branches;
//
//             // Create DOWN branch (branchVar <= down)
//             downChild = context.makeBranch(branchVar, down, IloCplex::BranchDown, obj);
//             ++branches;
//
//             // We don't use the unique ids of the down and up child. We only
//             // have them so illustrate how they are returned from
//             // CPXcallbackmakebranch().
//             (void)downChild;
//             (void)upChild;
//         }
//         v.end();
//     }
//
//     int getCalls() const { return calls; }
//     int getBranches() const { return branches; }
// };

void populate(IloModel model, NumVarMatrix edgeUsage, vector<vector<int>> *vertexMatrix, vector<vector<int>> *costMatrix, int *capacity)
{
    int n = vertexMatrix->size();
    int sumOfDemands = 0;
    for (int i = 1; i < n; i++)
    {
        sumOfDemands += vertexMatrix->at(i).at(2);
    }

    int kMin = 1 + sumOfDemands / *capacity;
    cout << "sumOfDemands: " << sumOfDemands << endl; // The lower bound for the number of vehicles needed
    cout << "capacity: " << *capacity << endl;        // The lower bound for the number of vehicles needed
    cout << "KMin: " << kMin << endl;                 // The lower bound for the number of vehicles needed

    IloEnv env = model.getEnv();
    IloObjective obj = IloMinimize(env);

    // Create all the edge variables
    edgeUsage[0] = IloNumVarArray(env, n, 0, 2, ILOINT); // Edges to the depot mmay be travelled twice, if the route is one costumer
    edgeUsage[0][0].setBounds(0, 0);                     // Cant travel from vertex i to vertex i

    for (int i = 1; i < n; i++)
    {
        edgeUsage[i] = IloNumVarArray(env, n, 0, 1, ILOINT);
        edgeUsage[i][i].setBounds(0, 0); // Cant travel from vertex i to vertex i
        edgeUsage[i][0].setBounds(0, 2); // Edges from the depot can be traveled twice
    }

    // Each vertex (except the depot) must have degree 2
    for (IloInt i = 1; i < n; ++i)
    {
        model.add(IloSum(edgeUsage[i]) == 2); // i.e. for each row the sum of edge Usage must be two
    }
    model.add(IloSum(edgeUsage[0]) >= 2 * kMin); // The depot must have degree of at least 2 kMin

    IloExpr v(env);
    for (IloInt i = 0; i < n; ++i)
    { // first column must be larger than 2*Kmin
        v += edgeUsage[i][0];
    }
    model.add(v >= 2 * kMin);
    v.end();

    for (IloInt j = 1; j < n; ++j)
    {
        IloExpr v(env);
        for (IloInt i = 0; i < n; ++i)
        { // all columns muust be equal to two as well
            v += edgeUsage[i][j];
        }
        model.add(v == 2);
        v.end();
    }

    for (IloInt j = 1; j < n; j++)
    {

        for (IloInt i = (j + 1); i < n; i++)
        {
            IloExpr v(env);
            v += edgeUsage[i][j];
            v -= edgeUsage[j][i];
            model.add(v == 0); // Forcing symmetry
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

    model.add(obj);
}

vector<vector<int>> createCPLEXInstance(vector<vector<int>> vertexMatrix, vector<vector<int>> costMatrix, int capacity)
{
    cout << "Creating instances!" << endl;

    IloEnv env;
    IloModel model(env);
    int n = vertexMatrix.size();
    NumVarMatrix edgeUsage(env, n);
    // typedef NumVarMatrix edgeCost(env, n);
    // IloRangeArray c(env);

    populate(model, edgeUsage, &vertexMatrix, &costMatrix, &capacity);

    IloCplex cplex(model);
    cplex.setParam(IloCplex::Param::TimeLimit, 60);

    // Use later to add custom branching
    // BranchCallback cb(x);
    // cplex.use(&cb, IloCplex::Callback::Context::Id::Branching);

    // Optimize the problem and obtain solution.
    if (!cplex.solve())
    {
        env.error() << "Failed to optimize LP" << endl;
    }

    NumVarMatrix vals(env);
    env.out() << "Solution status = " << cplex.getStatus() << endl;
    env.out() << "Solution value  = " << cplex.getObjValue() << endl;
    int value;
    vector<vector<int>> edgeUsageSolution(n, vector<int>(n,0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            value = cplex.getValue(edgeUsage[i][j]);
            cout << value << "  ";
            edgeUsageSolution.at(i).at(j) = value;
        }
        cout << "\n";
    }

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
    return edgeUsageSolution;
}

int main()
{
    CVRPGrapher grapher;

    tuple<vector<vector<int>>, int> instance = generateCVRPInstance(10, "C", "C", 0);
    vector<vector<int>> customers = get<0>(instance);
    int capacity = get<1>(instance);
    vector<vector<int>> costVector = calculateEdgeCost(&customers);
    
    grapher.setInstanceCoordinates(customers);
    
    printVector(customers);
    printVector(costVector);
    
    vector<vector<int>> edgeUsage = createCPLEXInstance(customers, costVector, capacity);
    vector<vector<int>> solution = fromEdgeUsageToRouteSolution(edgeUsage);
    
    printVector(solution);
    
    grapher.setSolutionVector(solution);



    return 0;
}
