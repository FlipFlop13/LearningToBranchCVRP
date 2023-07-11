#include "./CVRPGrapher.h"
#include <iostream>
#include "./utilities.h"
#include <unistd.h>


using namespace std;

CVRPGrapher::CVRPGrapher()
{
  fprintf(gnuplotPipe, "set title 'CVRP' \n ");
  fprintf(gnuplotPipe, "set term wxt title 'Capacitated Vehicle Routing Problem' \n ");
  fprintf(gnuplotPipe, "set xrange [1:1000]\n ");
  fprintf(gnuplotPipe, "set yrange [1:1000]\n ");
  fprintf(gnuplotPipe, "set palette maxcolors 2 \n ");
  fprintf(gnuplotPipe, "set palette defined ( 0 'blue', 1 'red') \n ");

  fprintf(gnuplotPipe, "unset colorbox \n ");
  fflush(gnuplotPipe);
}
CVRPGrapher::~CVRPGrapher()
{
  fprintf(gnuplotPipe, " quit \n ");
  fflush(gnuplotPipe);
  pclose(gnuplotPipe);
}


/// @brief Takes a 2D integer vector and saves it as a space separated txt file to be used by gnuplot, using defaultFilename. IT saves the depot coordinates in a new file.
/// @param vec Coordinate vector.
void CVRPGrapher::vec2File()
{

  ofstream fw(defaultFilename, ofstream::out);
  ofstream fwD(defaultDepotFilename, ofstream::out);

  int length = instanceCoordinates.size();
  if (fw.is_open())
  {
    // store array contents to text file
    for (int i = 0; i < length; i++)
    {
      if (i == 0)
      {
        fwD << instanceCoordinates[i][0] << " " << instanceCoordinates[i][1] << " \n";

        continue;
      }

      fw << instanceCoordinates[i][0] << " " << instanceCoordinates[i][1] << " \n";
    }
    fw.close();
  }
  else
    cout << "Problem with opening file";
}

void CVRPGrapher::plotCurrentInstance()
{
  if (gnuplotPipe)
  {
    fflush(gnuplotPipe);
    fprintf(gnuplotPipe, "unset arrow \n");

    string c = "plot '" + defaultFilename + "'  ps 1  lc rgb 'blue' notitle, '" + defaultDepotFilename + "'  ps 1 pt 7 lc rgb 'red' notitle\n ";
    cout << c;
    const char *gnuplotCommand = c.c_str();
    fprintf(gnuplotPipe, gnuplotCommand);
    fprintf(gnuplotPipe, "pause -1 \n ");
    fprintf(gnuplotPipe, "system 'clear'\n ");
    fflush(gnuplotPipe);
  }
}
void CVRPGrapher::replot()
{
  string c = string("set title 'CVRP Cost:") + to_string(instanceCost) + "'\n ";
  const char *gnuplotCommand = c.c_str();
  fprintf(gnuplotPipe, gnuplotCommand);
  fprintf(gnuplotPipe, "replot \n ");
  fprintf(gnuplotPipe, "pause -1 \n ");
  fflush(gnuplotPipe);
}

void CVRPGrapher::plotSolution()
{
  fflush(gnuplotPipe);

  fprintf(gnuplotPipe, "unset arrow \n"); // remove all the arrows that are in the current figure
  int x0, y0, x1, y1;
  vector<string> colorVector = {"blue", "dark-grey", "red", "web-green", "web-blue", "dark-cyan", "purple", "dark-red", "dark-chartreuse", "yellow", "turquoise", "light-red", "light-green", "light-blue"};
  int colorIdx = 0;
  string lineColor = "blue";
  for (vector<int> route : solutionVector)
  {
    int routeLength = route.size();
    for (int i = 0; i < (routeLength - 1); i++)
    {
      x0 = instanceCoordinates[route[i]][0];
      y0 = instanceCoordinates[route[i]][1];
      x1 = instanceCoordinates[route[i + 1]][0];
      y1 = instanceCoordinates[route[i + 1]][1];

      lineColor = colorVector[colorIdx];

      string c = string("set arrow from ") + to_string(x0) + ", " + to_string(y0) + " to " + to_string(x1) + ", " + to_string(y1) + " lc rgb '" + lineColor + "' nohead \n ";
      const char *gnuplotCommand = c.c_str();
      fprintf(gnuplotPipe, gnuplotCommand);
    }
    colorIdx++;
    if (colorIdx == colorVector.size())
    {
      colorIdx = 0;
    }
  }
  fprintf(gnuplotPipe, "replot \n ");
  fprintf(gnuplotPipe, "pause -1 \n ");
  fprintf(gnuplotPipe, "system 'clear'\n ");
  fflush(gnuplotPipe);
  replot();
}

void CVRPGrapher::setInstanceCoordinates(vector<vector<int>> vec)
{
  instanceCoordinates = vec;
  vec2File();
  plotCurrentInstance();
}

void CVRPGrapher::setInstanceCost(int cost)
{
  instanceCost = cost;
}

void CVRPGrapher::setSolutionVector(vector<vector<int>> vec)
{
  solutionVector = vec;
  plotSolution();
}

// int main(int argc, char **argv)
// {
//   CVRPGrapher grapher;
//   vector<vector<int>> vec;
//   tuple<vector<vector<int>>, int> instance;
//   tuple<vector<vector<int>>, int> solution;
//   // vec = generateCustomerCoordinates(1000, "R");
//   // grapher.vec2Graph(vec);
//   // sleep(2);
//   //
//   // vec = generateCustomerCoordinates(1000, "C");
//   // grapher.vec2Graph(vec);
//   // sleep(2);
//   //
//   // vec = generateCustomerCoordinates(1000, "RC");
//   // grapher.vec2Graph(vec);
//   // sleep(2);
//   //
//   // vec = generateCustomerCoordinates(1000, "RC");
//   // grapher.vec2Graph(vec);
//   // sleep(2);
//   // grapher.addLines();

//   // instance = readCVRP("./Graphs/X/X-n125-k30.vrp");
//   // solution = readSolution("./Graphs/X/X-n125-k30.sol");

//   // grapher.setInstanceCoordinates(get<0>(instance));
//   // grapher.setSolutionVector(get<0>(solution));
//   // grapher.setInstanceCost(get<1>(solution));

//   // instance = readCVRP("./Graphs/X/X-n148-k46.vrp");
//   // solution = readSolution("./Graphs/X/X-n148-k46.sol");
//   // sleep(3);
//   // grapher.setInstanceCoordinates(get<0>(instance));
//   // grapher.setSolutionVector(get<0>(solution));
//   // grapher.setInstanceCost(get<1>(solution));

//   cout << "looking for files" << endl;
//   vector<string> vrpInstances = glob("./Graphs/X/", ".vrp");
//   vector<string> vrpSolutions = glob("./Graphs/X/", ".sol");
//   printVector(vrpInstances);
//   printVector(vrpSolutions);
//   int nFiles = vrpInstances.size();
//   for (int i = 0; i < nFiles; i++)
//   {
//     instance = readCVRP(vrpInstances.at(i));
//     solution = readSolution(vrpSolutions.at(i));
//     sleep(2);
//     grapher.setInstanceCoordinates(get<0>(instance));
//     grapher.setSolutionVector(get<0>(solution));
//     grapher.setInstanceCost(get<1>(solution));
//   }

//   return 0;
// }

//int main(){
//return 0;
//}
