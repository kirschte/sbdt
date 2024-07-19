#include <iostream>
#include <algorithm>
#include "rdp_accountant.h"

using namespace std;

/** Constructors */

RDPAccountant::RDPAccountant(ModelParams *parameters, TreeParams *tree_parameters) : params(parameters), tree_params(tree_parameters)
{
}
    
RDPAccountant::~RDPAccountant() {
}

double RDPAccountant::gen_individual_sens(double gradient_length) {
    return gradient_length;
}

double RDPAccountant::gen_rho(double individual_sens, double hess_individual_sens) {
    return gen_rho(tree_params->rdp_alpha, tree_params->active_noise_scale, individual_sens, hess_individual_sens);
}

double RDPAccountant::cached_factor(double alpha, double j, double Q) {
    // Check if precomputed
    std::map<std::tuple<const double, const double, const double>, const double>::iterator lookup = tree_params->factor_table.find(std::tuple<const double, const double, const double>(alpha, j, Q));
    if (lookup != tree_params->factor_table.end()) {
        return lookup->second;
    }

    // Tight privacy amplification
    double b = 0.0;
    double f = 0.0;
    if (j < alpha) {
        b = binom(alpha, j);
        f = b + std::log(1.-Q) * (alpha - j) + std::log(Q) * j;
    } else {
        f = std::log(Q) * j;
    }
    if (b > std::numeric_limits<double>::max() || b < -std::numeric_limits<double>::max()
    || f > std::numeric_limits<double>::max() || f < -std::numeric_limits<double>::max()) {
        std::cerr << "Binomial coefficient b=(" << alpha << " " << j << ")=" << std::exp(b) << " or f=" << f << " is inf." << std::endl;
        std::cerr << "Q=" << Q << std::endl;
        std::cerr << "log(Q)=" << std::log(Q) << " log(1-Q)=" << std::log(1.-Q) << " log(b)=" << b << std::endl;
        std::exit(EXIT_FAILURE);

        // Binomial coefficient is too large. Approximating it (Stirling's approximation). Deactivated for now, as this introduces an error.
        double approx_log_b = (alpha+0.5)*std::log(alpha) - (j+0.5)*std::log(j) - (alpha - j + 0.5)*std::log(alpha-j) - 0.5*std::log(2.*M_PI);
        f = approx_log_b + std::log(1-Q) * (alpha - j) + std::log(Q) * j;
        if (f > std::numeric_limits<double>::max() || f < -std::numeric_limits<double>::max()) {
            std::cerr << "Factor is inf." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    
    // Cache for later use
    tree_params->factor_table.insert(std::pair<std::tuple<const double, const double, const double>, const double>(std::tuple<double, double, double>(alpha, j, Q), f));
    return f;
}

double RDPAccountant::gen_rho(double alpha, double noise_scale, double individual_sens, double hess_individual_sens) {

    if (tree_params->active_subsampling_ratio < 1.0) {
        // Check if precomputed
        std::map<std::tuple<const double, const double, const double, const double>, const double>::iterator lookup = tree_params->leaf_rho_table.find(
            std::tuple<const double, const double, const double, const double>(alpha, tree_params->active_subsampling_ratio, noise_scale, individual_sens));
        if (lookup != tree_params->leaf_rho_table.end()) {
            return lookup->second;
        }
    }
    
    double rho_log = 0.0;
    double sigma = noise_scale;
    const double sigma_pow = std::pow(sigma, 2.);
    const double individual_sens_pow = std::pow(individual_sens, 2.);
    const double sens_pow = std::pow(tree_params->active_threshold, 2.);
    const double hess_individual_sens_pow = std::pow(hess_individual_sens, 2.);
    const double hess_sens_pow = params->newton_boosting ? std::pow(tree_params->hess_active_threshold, 2.) : 1.0;
    const double r = params->leaf_denom_noise_weight;
    const double dim_half = fpclassify(r) == FP_ZERO ? 0.5 : 1; // number of dimensions halved
    const double mu_omega = dim_half * ( ((1-r) * individual_sens_pow) / (sens_pow * sigma_pow) 
        + r * hess_individual_sens_pow / (hess_sens_pow * sigma_pow));
    double leaf_rho = 0.0;
    if (tree_params->active_subsampling_ratio < 1.0) {
        std::vector<double> log_exp(alpha, 0.0);
        log_exp[0] = std::log(1-tree_params->active_subsampling_ratio) * (alpha-1.) + std::log(alpha*tree_params->active_subsampling_ratio - tree_params->active_subsampling_ratio + 1.);
        int cnt = 1;
        for (int j=2; j<=alpha; j++) {
            // Returns log of factor
            double log_factor = cached_factor(alpha, static_cast<double>(j), tree_params->active_subsampling_ratio);
            log_exp[cnt] = log_factor + (j - 1.0) * (j * mu_omega);
            if (log_exp[cnt] > std::numeric_limits<double>::max() || log_exp[cnt] < -std::numeric_limits<double>::max()) {
                std::cerr << "Failure in Renyi filter: value in logsumexp is inf." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            cnt += 1;
        }

        rho_log = log_sum_exp(log_exp);

        if (rho_log > std::numeric_limits<double>::max() || rho_log < -std::numeric_limits<double>::max()) {
            std::cerr << "Failure in Renyi filter: value in rho_log is inf." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        leaf_rho = 1./(alpha-1) * rho_log;
        
        if(leaf_rho < - 10e-10 || rho_log < - 10e-10) {
            std::cout << "alpha=" << alpha << " Q=" << tree_params->active_subsampling_ratio << " sigma=" << sigma << std::endl;
            std::cout << "leaf_rho=" << leaf_rho << std::endl;
            std::cout << "rho_log=" << rho_log << std::endl;
            std::cout << "binom=" << binom(alpha, 1000.) << std::endl;
            
            std::exit(EXIT_FAILURE);
        }
    } else {
        leaf_rho = alpha * mu_omega;
    }

    if (tree_params->active_subsampling_ratio < 1.0) {
        // Cache for later use
        tree_params->leaf_rho_table.insert(std::pair<std::tuple<const double, const double, const double, const double>, const double>(
            std::tuple<double, double, double, double>(alpha, tree_params->active_subsampling_ratio, noise_scale, individual_sens), leaf_rho));
    }
    return leaf_rho;
};

double RDPAccountant::noise_scale_guess() {
    return sqrt(2.0*log(1.25/DELTA))*sqrt(params->nb_trees)*tree_params->active_subsampling_ratio 
        * 1 / tree_params->leaf_eps;
}

void RDPAccountant::setup_approximation() {
    tree_params->rho_approx_table.clear();
    const int num_steps = 1000.0;
    const int hess_num_steps = params->newton_boosting ? 1000 : 1;
    const double max_gradient_length = std::max(params->l2_threshold, params->pf_l2_threshold);
    const double max_hess_length = std::max(params->hess_l2_threshold, params->pf_hess_l2_threshold);
    for (int i=0; i<num_steps; i++) {
        for (int j=0; j<hess_num_steps; j++) {
            const double gradient_length = (i+1.) / num_steps * max_gradient_length;
            const double hess_length = params->newton_boosting ? (j+1.) / num_steps * max_hess_length : (1.0 + 10e-8);
            const double rho = gen_rho(gradient_length, hess_length);
            tree_params->rho_approx_table.insert(std::pair<std::pair<const double, const double>, const double>(std::pair<const double, const double>(gradient_length, hess_length), rho));
        }
    }
    double rho = gen_rho(max_gradient_length+10e-8, max_hess_length+10e-8);
    tree_params->rho_approx_table.insert(std::pair<std::pair<const double, const double>, const double>(std::pair<const double, const double>(max_gradient_length+10e-8, max_hess_length+10e-8), rho));
}

double RDPAccountant::approximate_rho(double individual_sens, double hess_individual_sens) {
    double rho = (tree_params->rho_approx_table.upper_bound(std::pair<const double, const double>(individual_sens, hess_individual_sens)))->second;
    return rho;
}

double RDPAccountant::calc_refine_splits_rho(double alpha, double noise_scale, double hess_sens) {
    const double temp = tree_params->active_subsampling_ratio;
    tree_params->active_subsampling_ratio = params->refine_splits_subsample;
    const double refine_splits_rho = gen_rho(alpha, noise_scale, 0.0, hess_sens) * params->refine_splits_rounds;
    tree_params->active_subsampling_ratio = temp;
    return refine_splits_rho;        
};

Accounting RDPAccountant::setup_accounting(int nb_compositions) {
    Accounting accounting_best = {.alpha = 0, .eps = 10e10, .delta = DELTA, .max_rho = 0, .noise_scale = 10e10};
    const double h = 1;  // derivative step size
    // Zhu et al. bound is valid for alpha >= 2.
    const double min_alpha = 2., max_alpha = MAX_ALPHA;
    const int max_steps = 30;
    const double target_eps = tree_params->leaf_eps;
    const double sens = tree_params->active_threshold;
    const double hess_sens = params->newton_boosting ? tree_params->hess_active_threshold : 1.0;
    double noise_scale = this->noise_scale_guess();
    double noise_scale_ub = 1000. * noise_scale;
    double noise_scale_lb = 10e-10;
    auto calc_eps = [this, sens, hess_sens, nb_compositions](double alpha, double noise_scale) { 
            double rho = gen_rho(alpha, noise_scale, sens, hess_sens) * nb_compositions;
            if (params->refine_splits) {
                rho += calc_refine_splits_rho(alpha, noise_scale, hess_sens);
            }
            const double eps = rho - (std::log(DELTA) + std::log(alpha+0.)) / (alpha - 1.) + std::log((alpha - 1.) / alpha);
            return eps;
    };
    double alpha = clamp(10, min_alpha+h, max_alpha-h);  // initial guess, consecutively updated
    for (int cnt=0; cnt<20; cnt++) {
        Accounting accounting = {.alpha = 0, .eps = 10e10, .delta = DELTA, .max_rho = 0, .noise_scale = 10e10};
        alpha = clamp(10, min_alpha+h, max_alpha-h);
        auto calc_eps_ = [this, sens, noise_scale, calc_eps](double alpha) { return calc_eps(alpha, noise_scale); };
        int step = 0;
        for (; step<max_steps; ++step) {
            const double alpha_rnded = std::round(alpha);
            // Newton step with finite differences approximation of jac/hess
            const double old_eps_minus_h = calc_eps_(alpha_rnded - h);
            const double old_eps = calc_eps_(alpha_rnded);
            const double old_eps_plus_h = calc_eps_(alpha_rnded + h);
            const double new_alpha = clamp(alpha - h * ( old_eps_plus_h - old_eps ) / ( old_eps_plus_h - 2*old_eps + old_eps_minus_h ), min_alpha+h, max_alpha-h );
            const double new_alpha_rnded = std::round(new_alpha);
            const double eps = calc_eps_(new_alpha_rnded);
            
            if (eps < accounting.eps) {
                accounting.eps = eps;
                accounting.alpha = new_alpha_rnded;
                accounting.max_rho = gen_rho(new_alpha_rnded, noise_scale, sens, hess_sens) * nb_compositions;
                if (params->refine_splits) {
                    accounting.max_rho += calc_refine_splits_rho(new_alpha_rnded, noise_scale, hess_sens);
                }
                accounting.noise_scale = noise_scale;
            }
            if (std::fabs(new_alpha / alpha - 1.0) <= 0.005 ) break;
            alpha = new_alpha;
        }
        if (step == max_steps) {  // fall-back to linear search
            for (alpha=min_alpha; alpha<=max_alpha; ++alpha) {
                const double eps = calc_eps_(alpha);

                if (eps < accounting.eps) {
                    accounting.eps = eps;
                    accounting.alpha = alpha;
                    accounting.max_rho = gen_rho(alpha, noise_scale, sens, hess_sens) * nb_compositions;
                    if (params->refine_splits) {
                        accounting.max_rho += calc_refine_splits_rho(alpha, noise_scale, hess_sens);
                    }
                    accounting.noise_scale = noise_scale;
                }
                if (accounting.alpha + 4 <= alpha) break;
            }
        }

        if (accounting.eps > target_eps) {
            noise_scale_lb = noise_scale;
        }
        if (accounting.eps < target_eps) {
            noise_scale_ub = noise_scale;
            if (accounting.noise_scale < accounting_best.noise_scale) {
                accounting_best = accounting;
            }
        }
        noise_scale = 0.5 * noise_scale_lb + 0.5 * noise_scale_ub;
    }

    // fallback to linear search if no sufficient result is calculated
    if (accounting_best.noise_scale == 10e10 || accounting_best.eps > target_eps || accounting_best.eps / target_eps < 0.9) {
        Accounting accounting_best = {.alpha = 0, .eps = 10e10, .delta = DELTA, .max_rho = 0, .noise_scale = 10e10};
        double noise_scale = this->noise_scale_guess();
        double noise_scale_ub = 1000. * noise_scale;
        double noise_scale_lb = 10e-10;
        for (int cnt=0; cnt<20; cnt++) {
            Accounting accounting = {.alpha = 0, .eps = 10e10, .delta = DELTA, .max_rho = 0, .noise_scale = 10e10};
            // Zhu et al. bound is valid for alpha >= 2.
            for (double alpha=min_alpha; alpha<=max_alpha; alpha++) {
                double rho = gen_rho(alpha, noise_scale, sens, hess_sens) * nb_compositions;
                if (params->refine_splits) {
                    rho += calc_refine_splits_rho(alpha, noise_scale, hess_sens);
                }
                const double eps = rho - (std::log(DELTA) + std::log(alpha+0.)) / (alpha - 1.) + std::log((alpha - 1.) / alpha);
                if (eps < accounting.eps) {
                    accounting.eps = eps;
                    accounting.alpha = alpha;
                    accounting.max_rho = rho;
                    accounting.noise_scale = noise_scale;
                }
                // there is only one optimal alpha thus if eps does not get lower we can halt early.
                // 4 is a safety margin for numerical stability
                if (accounting.alpha + 4 <= alpha) break;
            }
            if (accounting.eps > target_eps) {
                noise_scale_lb = noise_scale;
            }
            if (accounting.eps < target_eps) {
                noise_scale_ub = noise_scale;
                if (accounting.noise_scale < accounting_best.noise_scale) {
                    accounting_best = accounting;
                }
            }
            noise_scale = 0.5 * noise_scale_lb + 0.5 * noise_scale_ub;
        }


        if (accounting_best.noise_scale == 10e10 || accounting_best.eps > target_eps || accounting_best.eps / target_eps < 0.9) {
            std::cout << "Warning: RDP accountant could not find optimal noise scale." << std::endl;
        }
        return accounting_best;
    }
    return accounting_best;
}

Accounting RDPAccountant::setup_accounting(int nb_compositions, double noise_scale) {
    Accounting accounting = {.alpha = 0, .eps = 10e10, .delta = DELTA, .max_rho = 0, .noise_scale = noise_scale};
    const double min_alpha = 2., max_alpha = MAX_ALPHA;
    const double sens = tree_params->active_threshold;
    const double hess_sens = params->newton_boosting ? tree_params->hess_active_threshold : 1.0;
    // Zhu et al. bound is valid for alpha >= 2.
    for (double alpha=min_alpha; alpha<max_alpha; alpha++) {
        double rho = gen_rho(alpha, noise_scale, sens, hess_sens) * nb_compositions;
        if (params->refine_splits) {
            rho += calc_refine_splits_rho(alpha, noise_scale, hess_sens);
        }
        const double eps = rho - (std::log(DELTA) + std::log(alpha+0.)) / (alpha - 1) + std::log((alpha - 1.) / alpha);
        if (eps < accounting.eps) {
            accounting.eps = eps;
            accounting.alpha = alpha;
            accounting.max_rho = rho;
        }
        // there is only one optimal alpha thus if eps does not get lower we can halt early.
        // 4 is a safety margin for numerical stability
        if (accounting.alpha + 4 <= alpha) break;
    }
    
    if (accounting.noise_scale >= 10e10) {
        std::cout << "Warning: RDP accountant returned very large noise scale. This may be due to an error in setup accounting." << std::endl;
    }
    return accounting;
}

double RDPAccountant::log_sum_exp(std::vector<double> &log_exp) {
    double max_elem = *std::max_element(log_exp.begin(), log_exp.end());
    double sum = std::accumulate(log_exp.begin(), log_exp.end(), 0.0,
     [max_elem](double a, double b) { return a + exp(b - max_elem); });
  return max_elem + std::log(sum);
}

double RDPAccountant::binom(double n, double k) {
    return
        (        k> n  )? std::log(0) :          // out of range
        (k==0 || k==n  )? std::log(1) :          // edge
        (k==1 || k==n-1)? std::log(n) :          // first
        binom(n - 1, k - 1) + std::log(n / k);   // recursive
}
