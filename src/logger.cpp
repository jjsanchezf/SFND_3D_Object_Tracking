
#include <fstream>
#include "logger.hpp"

using namespace std;

ofstream outFile("OutputData.txt");

void logger(std::string logMsg)
{
    outFile << logMsg;
}
