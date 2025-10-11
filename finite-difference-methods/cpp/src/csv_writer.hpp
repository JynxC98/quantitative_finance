#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include <fstream>
#include <vector>
#include <string>

class CSVWriter
{
private:
    std::ofstream file;

public:
    CSVWriter(const std::string &filename);
    ~CSVWriter();

    void writeRow(const std::vector<double> &row);
    void writeRow(const std::vector<std::string> &row);
};

#endif