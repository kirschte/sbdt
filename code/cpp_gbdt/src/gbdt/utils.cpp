#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <mutex>
#include <iostream>
#include "utils.h"

/** Global Variables */

bool AVGLEAKAGE_MODE;
size_t cv_fold_index;
std::once_flag flag_dp, flag_minsampl, flag_featbord;


/** Methods */

// create some default parameters for quick testing
ModelParams create_default_params()
{
    ModelParams params;
    return params;
};


// put a value between two bounds, not in std::algorithm in c++11
double clamp(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}


double log_sum_exp(std::vector<double> vec)
{
    size_t count = vec.size();
    if (count > 0) {
        double maxVal = *std::max_element(vec.begin(), vec.end());
        double sum = 0;
        for (size_t i = 0; i < count; i++) {
            sum += exp(vec[i] - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
}

double compute_mean(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double compute_stdev(std::vector<double> &vec, double mean)
{
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    return std::sqrt(sq_sum / vec.size() - mean * mean);
}


std::string get_time_string()
{
    time_t t = time(0);
    struct tm *now = localtime(&t);
    char buffer [80];
    strftime(buffer,80,"%m.%d_%H:%M:%S",now);
    return std::string(buffer);
}

// Fisher–Yates shuffle
template<class T>
void random_unique(std::vector<T> &a, size_t num) {
    if (a.size() > num) {
        for(size_t i = a.size() - 1; i >= a.size() - num; --i) {
            // note: rand()+mod is not perfectly uniformly distributed but 3x faster
            int r = std::rand() % ( i + 1 );
            std::swap(a[i], a[r]);
        }
        a.erase(a.begin(), a.end()-num);
    }
}

template void random_unique<int>(std::vector<int>&, size_t);
template void random_unique<double>(std::vector<double>&, size_t);
