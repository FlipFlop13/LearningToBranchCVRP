#include <iostream>
#include "./utilities.h"
#include <unistd.h>

using namespace std;

class Grapher
{
  // Private Variables
  FILE *gnuplotPipe = popen("gnuplot -persist", "w"); // Open a pipe to gnuplot
  string defaultFilename = "./graphingCache/temp.txt";
  // Public Variables
public:
  // constructor
  Grapher()
  {
    fprintf(gnuplotPipe, "set title 'CVRP' \n ");
    fprintf(gnuplotPipe, "set xrange [1:1000]\n ");
    fprintf(gnuplotPipe, "set yrange [1:1000]\n ");
    fprintf(gnuplotPipe, "set palette maxcolors 2 \n ");
    fprintf(gnuplotPipe, "set palette defined ( 0 'blue', 1 'red') \n ");

    fprintf(gnuplotPipe, "unset colorbox \n ");
    fflush(gnuplotPipe);
  }
  ~Grapher()
  {
    fprintf(gnuplotPipe, " quit \n ");
    fflush(gnuplotPipe);
    pclose(gnuplotPipe);
  }

  // Functions
  void vec2File(vector<vector<int>> vec);
  void vec2File(vector<vector<int>> vec, string filename);
  void vec2Graph(vector<vector<int>> vec);
  void addLines();
};

/// @brief Takes a 2D integer vector and saves it as a space separated txt file to be used by gnuplot.
/// @param vec
/// @param filename
void Grapher::vec2File(vector<vector<int>> vec, string filename)
{
  cout << "placing in file" << endl;
  ofstream fw(filename, ofstream::out);

  if (fw.is_open())
  {
    // store array contents to text file
    for (vector<int> line : vec)
    {
      fw << line[0] << " " << line[1] << "\n";
    }
    fw.close();
  }
  else
    cout << "Problem with opening file";
}
/// @brief Takes a 2D integer vector and saves it as a space separated txt file to be used by gnuplot, using defaultFilename.
/// @param vec
void Grapher::vec2File(vector<vector<int>> vec)
{
  string filename = defaultFilename;
  cout << "placing in file" << endl;
  cout << filename << endl;

  ofstream fw(filename, ofstream::out);

  int length = vec.size();
  if (fw.is_open())
  {
    // store array contents to text file
    for (int i = 0; i < length; i++)
    {
      if (i == 0)
      {
        fw << vec[i][0] << " " << vec[i][1] << " "
           << "1"
           << "\n";

        continue;
      }

      fw << vec[i][0] << " " << vec[i][1] << " "
         << "0"
         << "\n";
    }
    fw.close();
  }
  else
    cout << "Problem with opening file";
}

void Grapher::vec2Graph(vector<vector<int>> vec)
{

  vec2File(vec);
  if (gnuplotPipe)
  {
    cout << "plotting" << endl;
    
    string c = "plot '" + defaultFilename + "'  with points palette notitle, './graphingCache/temp2.txt'  ps 1 pt 7 lc rgb 'red' notitle\n ";
    const char *gnuplotCommand = c.c_str();
    cout << c << endl;
    fprintf(gnuplotPipe, gnuplotCommand);
    fprintf(gnuplotPipe, "pause -1 \n ");
    fflush(gnuplotPipe);
  }
}

void Grapher::addLines(){

fprintf(gnuplotPipe, "set arrow 2 from 500,400 to 800,900 nohead \n ");
fprintf(gnuplotPipe, "set arrow 3 from 600,400 to 700,900 nohead \n ");
fprintf(gnuplotPipe, "set arrow 4 from 700,400 to 600,900 nohead \n ");
fprintf(gnuplotPipe, "replot \n ");
fprintf(gnuplotPipe, "pause -1 \n ");

fflush(gnuplotPipe);
}


int main(int argc, char **argv)
{
  Grapher grapher;
  vector<vector<int>> vec;
  // vec = generateCustomerCoordinates(1000, "R");
  // grapher.vec2Graph(vec);
  // sleep(2);

  // vec = generateCustomerCoordinates(1000, "C");
  // grapher.vec2Graph(vec);
  // sleep(2);

  // vec = generateCustomerCoordinates(1000, "RC");
  // grapher.vec2Graph(vec);
  // sleep(2);

  vec = generateCustomerCoordinates(1000, "RC");
  grapher.vec2Graph(vec);
  sleep(2);
  grapher.addLines();
    // grapher.vec2Graph(vec);

  sleep(100);
  return 0;
}
