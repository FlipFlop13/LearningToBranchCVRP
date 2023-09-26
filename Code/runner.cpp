#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


int main()
{
    std::map<int, std::vector<int>> loadedData;
    // std::map<int, int> loadedData;
    std::ifstream inputFile("./bin/training/Instance0/tree.dat");
    boost::archive::text_iarchive ia(inputFile);
    ia >> loadedData;

    for (auto i : loadedData)
    {
        std::cout << i.first << ": " << i.second.at(0) << ", " << i.second.at(1)
             << std::endl;
    }
    return 0;
}