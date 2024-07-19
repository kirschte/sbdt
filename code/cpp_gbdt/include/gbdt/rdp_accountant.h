#ifndef RDPACCOUNTANT_H
#define RDPACCOUNTANT_H

#include "parameters.h"
#include "utils.h"

class RDPAccountant
{
public:
    // constructors
    RDPAccountant(ModelParams *params, TreeParams *tree_params);
    ~RDPAccountant();

    // methods
    Accounting setup_accounting(int nb_compositions);
    Accounting setup_accounting(int nb_compositions, double noise_scale);
    void setup_approximation();
    double approximate_rho(double individual_sens, double individual_hess_sens);
    double calc_refine_splits_rho(double alpha, double noise_scale, double hess_sens);
    double noise_scale_guess();
    double gen_individual_sens(double gradient_length);
    double gen_rho(double alpha, double noise_scale, double individual_sens, double hess_individual_sens);
    double gen_rho(double individual_sens, double hess_individual_sens);
    double log_sum_exp(std::vector<double> &log_exp);
    double binom(double n, double k);
    double cached_factor(double alpha, double j, double Q);
    
private:
    // fields
    ModelParams *params;
    TreeParams *tree_params;
};

#endif // RDPACCOUNTANT_