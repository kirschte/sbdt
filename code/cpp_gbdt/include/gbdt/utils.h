#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include "parameters.h"

typedef std::vector<std::vector<double>> VVD;
typedef std::vector<std::vector<int>> VVI;


// method declarations
ModelParams create_default_params();
double clamp(double n, double lower, double upper);
double log_sum_exp(std::vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');
double compute_mean(std::vector<double> &vec);
double compute_stdev(std::vector<double> &vec, double mean);

std::string get_time_string();

template<class T>
void random_unique(std::vector<T> &a, size_t num);

#endif // UTILS_H