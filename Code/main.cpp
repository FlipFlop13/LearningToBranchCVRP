#include <iostream>
#include "./utilities.h"
using namespace std;


int main(){
    // tuple t = readCVRP("./Graphs/X/X-n110-k13.vrp");
    // writeCVRP(get<0>(t), get<1>(t));
    // generate();
    tuple sol = readSolution();
    writeSolution(get<0>(sol), get<1>(sol));
    return 0;
}