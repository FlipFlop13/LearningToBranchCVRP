int main()
{
    std::map jsonMap = readJson();
    std::map<std::string, string>::iterator it = jsonMap.begin();
    for (auto i : jsonMap)
    {
        cout << i.first << " " << i.second
             << endl;
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
    streambuf *coutBufBack = std::cout.rdbuf();
    streambuf *coutbuf = out.rdbuf();
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
    tie(edgeUsage, cost) = createAndRunCPLEXInstance(customers, costVector, capacity, true, false);
    cout << "Cost: " << cost << endl;
    // tie(edgeUsage, cost) = createAndRunCPLEXInstance(customers, costVector, capacity, true, true);
    // cout << "Cost: " << cost << endl;

    vector<vector<int>> solution = fromEdgeUsageToRouteSolution(edgeUsage);
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
