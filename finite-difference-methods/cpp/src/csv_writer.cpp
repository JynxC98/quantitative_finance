#include "csv_writer.hpp"
#include <iostream>

CSVWriter::CSVWriter(const std::string &filename)
{
    file.open(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

CSVWriter::~CSVWriter()
{
    if (file.is_open())
    {
        file.close();
    }
}

void CSVWriter::writeRow(const std::vector<double> &row)
{
    for (size_t i = 0; i < row.size(); ++i)
    {
        file << row[i];
        if (i != row.size() - 1)
        {
            file << ",";
        }
    }
    file << "\n";
}

void CSVWriter::writeRow(const std::vector<std::string> &row)
{
    for (size_t i = 0; i < row.size(); ++i)
    {
        file << row[i];
        if (i != row.size() - 1)
        {
            file << ",";
        }
    }
    file << "\n";
}