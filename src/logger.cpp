
#include <fstream>
#include "logger.hpp"

using namespace std;

ofstream outFile("../Output/OutputData.txt");

void logger(std::string logMsg)
{
    outFile << logMsg;
}
