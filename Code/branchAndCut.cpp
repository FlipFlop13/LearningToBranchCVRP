#include "./branchAndCut.h"
std::mutex mtxCVRP; // mutex for critical section

NodeData::NodeData(int a, int b, int c, int d, int e, int f, double g)
{
    nodeUID = a;
    childUp = b;
    childDown = c;
    branchedVariableI = d;
    branchedVariableJ = e;
    nodeDepth = f;
    relaxationObjective = g;
}
int NodeData::getUID() { return nodeUID; }

template <class Archive>
void NodeData::serialize(Archive &a, const unsigned version)
{
    a & nodeUID & childUp & childDown & branchedVariableI & branchedVariableJ & nodeDepth & relaxationObjective;
}

//----------------------------CALLBACK----------------------------
CVRPCallback::CVRPCallback(const IloInt capacity, const IloInt n, const vector<int> demands,
                           const NumVarMatrix &_edgeUsage, bool _training) : edgeUsage(_edgeUsage)
{
    training = _training;
    Q = capacity;
    N = n;
    demandVector = demands;
    vector<string> directories = glob("./bin/training/", "*");
    filepathCGC = "./bin/training/Instance" + to_string(size(directories));

    struct stat sb;
    if (!(stat(filepathCGC.c_str(), &sb) == 0))
    {
        mkdir(filepathCGC.c_str(), 0777);
    }
}

// This wil check the connected components, guarantueed to find an error (if one exists)
// for integer solutions (hence it is always called with a candidate mask)
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
CVRPCallback::branching1(const IloCplex::Callback::Context &context) const
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
        mtxCVRP.lock();
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
        mtxCVRP.unlock();
    }
}

inline void
CVRPCallback::branching2(const IloCplex::Callback::Context &context) const
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

void CVRPCallback::saveNodeInstance(NodeData data) const
{
    {
        string filename = filepathCGC + "/Node" + to_string(data.getUID()) + ".dat";
        std::ofstream outfile(filename);
        boost::archive::text_oarchive archive(outfile);
        archive << data;
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
map<int, vector<int>> CVRPCallback::getTree() const
{
    return tree;
}

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

//----------------------------Create CPLEX----------------------------



