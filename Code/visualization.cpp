#include <iostream>
// #include <gnuplot.h>
#include "./utilities.h"
using namespace std;

// /// @brief Creates a DAT file that is send to gnuplot to plot
// /// @param
// int createDATFile(vector<int> vec){
// return 0;

// }


void vec2Graph(vector<vector<int>> vec){
    ofstream fw("CPlusPlusSampleFile.txt", ofstream::out);

    if (fw.is_open())
{
  //store array contents to text file
  for (vector<int> line : vec) {
    fw << line[0] <<" " <<line[1] << "\n";
  }
  fw.close();
}
else cout << "Problem with opening file";

FILE *gnuplotPipe = popen("gnuplot -persist", "w"); // Open a pipe to gnuplot

    if (gnuplotPipe)
    {
        cout << "plotting" << endl;
        // fprintf(gnuplotPipe, "plot 'YYY.txt'\n");
        // fprintf(gnuplotPipe, "pause 2 \n");
        // fprintf(gnuplotPipe, "set arrow from 400,400 to 900,900 nohead lc rgb \'red\'\n");

        fprintf(gnuplotPipe, "plot 'CPlusPlusSampleFile.txt'\n");
        fflush(gnuplotPipe);

        fprintf(gnuplotPipe, "\n exit \n"); // exit gnuplot

        // pclose(gnuplotPipe);
    }

}






int main(int argc, char **argv)
{
    vector<vector<int>> vec = generateCustomerCoordinates(1000, "CR");
    vec2Graph(vec);
    return 0;
}
