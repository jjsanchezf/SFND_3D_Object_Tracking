#ifndef logger_hpp
#define logger_hpp

#include <fstream>
using namespace std;


void loggerOpen();
void loggerClose();
void logger(std::string logMsg);

#endif