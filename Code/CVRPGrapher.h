#ifndef CVRPGrapher_G // include guard
#define CVRPGrapher_G

#include <iostream>
#include <vector>
#include <unistd.h>


class CVRPGrapher
{
  // Private Variables
  FILE *gnuplotPipe = popen("gnuplot -persist", "w"); // Open a pipe to gnuplot
  std::string defaultFilename = "./graphingCache/temp.txt";
  std::string defaultDepotFilename = "./graphingCache/tempDepot.txt";
  // Public Variables
public:
  // constructor
  std::vector<std::vector<int>> instanceCoordinates;
  std::vector<std::vector<int>> solutionVector;
  int instanceCost = 0;
  CVRPGrapher();
  ~CVRPGrapher();


  // Functions
  void setInstanceCoordinates(std::vector<std::vector<int>> vec);
  void setSolutionVector(std::vector<std::vector<int>> vec);
  void setInstanceCost(int cost);
  void vec2File();
  void plotCurrentInstance();
  void plotSolution();
  void replot();
};

#endif /* CVRPGrapher_G */