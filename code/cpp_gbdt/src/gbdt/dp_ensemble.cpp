#include <numeric>
#include <algorithm>
#include <mutex>
#include <iostream>
#include "dp_ensemble.h"
#include "rdp_accountant.h"

extern size_t cv_fold_index;
extern std::once_flag flag_dp, flag_minsampl;

using namespace std;

/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    // only output this once, in case we're running with multiple threads
    if (parameters->privacy_budget == 0 or !parameters->use_dp){
        std::call_once(flag_dp, [](){std::cout << "!!! DP disabled !!! (slower than dp!)" << std::endl;});
        params->use_dp = false;
        params->privacy_budget = 0;
    }
    if (parameters->min_samples_split > 0 && parameters->use_dp) {
        std::call_once(flag_minsampl, [](){std::cout << "!!! Invalid DP guarantees !!! (better set min_samples_split=0)" << std::endl;});
    }
    if (params->gradient_filtering && params->leaf_noise == GAUSS) {
        std::cerr << "Gradient Filtering can not be enabled for Gauss noise." << std::endl;
    }
}
    
DPEnsemble::~DPEnsemble() {
    for (auto &tree : trees) {
        tree.delete_tree(tree.root_node);
    }
    delete this->tree_params;
}


/** Methods */

void DPEnsemble::train(DataSet *dset) {
    this->real_eps = params->privacy_budget;
    DataSet novel_dataset;
    DataSet filtered_dataset;
    this->dataset = dset;
    DataSet dataset_cpy = *dataset;
    int original_length = dataset->length;
  
    // set identifiers, so we can track privacy budget per datapoint correctly
    for (int i=0; i<original_length; i++) {
        dataset_cpy.identifiers.push_back(i);
    }

    // setup for poisson subsampling
    std::random_device rd;
    std::mt19937 gen(rd());

    // compute initial prediction (for continuous learning, set it later)
    if (!params->continuous_learning) {
        this->init_score = params->task->compute_init_score(dataset->y, params->init_score_threshold, params->use_dp, params->privacy_budget * params->privacy_budget_init_score_ratio);
        LOG_DEBUG("Training initialized with score: {1}", init_score);
    }
    
    // each tree gets the full pb, as they train on distinct data
    double per_ensemble_privacy_budget = params->privacy_budget * (1.0 - params->privacy_budget_init_score_ratio);

    // disable gain privacy budget if we determine the split randomly.
    double gain_privacy_share = ((params->max_feature_values == RAND) && (params->max_features == ONE || params->max_features == RAND))
            ? 0.0
            : params->privacy_budget_gain_ratio;
    if (this->tree_params != nullptr) delete this->tree_params;
    this->tree_params = new TreeParams(gain_privacy_share);
    tree_params->tree_privacy_budget = per_ensemble_privacy_budget;
    tree_params->active_threshold = params->l2_threshold;
    tree_params->hess_active_threshold = params->hess_l2_threshold;
    tree_params->active_subsampling_ratio = params->subsampling_ratio;
    tree_params->leaf_eps = (1.0 - tree_params->gain_privacy_share) * tree_params->tree_privacy_budget;
    // Setup cache storage for predictions
    tree_params->saved_predictions = std::vector<double>(dataset_cpy.length, 0.0);

    // setup feature weights
    std::vector<double> weights(dataset->num_x_cols, params->numeric_feature_weight);
    for (auto ix: params->cat_idx) {
        weights[ix] = params->cat_values[ix].size();
    }
    tree_params->feature_weights = weights;   

    // Setup RDP accountant
    RDPAccountant rdp_accountant = RDPAccountant(this->params, this->tree_params);
    if (params->leaf_noise == GAUSS) {
        tree_params->active_noise_scale = rdp_accountant.noise_scale_guess();
    }

    // Setup privacy filter    
    // for renyi accountant
    std::vector<double> accounted_rho;
    if (params->use_privacy_filter) {
        accounted_rho = std::vector<double>(original_length, 0.0);
        if (params->leaf_noise == LAPLACE) {
            std::cerr << "WARNING: Renyi Filter can only account for Gauss noise." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (params->refine_splits) {
            std::cerr << "WARNING: Renyi Filter accounting for rounds of refined splits is not yet implemented." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    double max_rho;
    if (params->leaf_noise == GAUSS and params->use_dp) {
        Accounting accounting;
        if (params->custom_noise_scale > 0.0) {
            // Use custom noise scale
            accounting = rdp_accountant.setup_accounting(params->custom_noise_scale);
            tree_params->active_noise_scale = params->custom_noise_scale;
        } else {
            // Finetune to optimal noise scale
            accounting = rdp_accountant.setup_accounting(params->nb_trees);
        }
        tree_params->rdp_alpha = accounting.alpha;
        tree_params->active_noise_scale = accounting.noise_scale;

        this->leaf_sigma = accounting.noise_scale;
        this->max_rho = accounting.max_rho;
        this->alpha = accounting.alpha;
        this->real_eps = params->privacy_budget + accounting.eps - tree_params->leaf_eps;
        max_rho = accounting.max_rho;

        if (accounting.delta > DELTA) {
            std::cerr << "Accounted delta is too large delta=" << accounting.delta << "." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (max_rho < 0.0) {
            std::cerr << "Failure in Renyi filter: max_rho/apx_n_max_rho is negative." << std::endl;
            std::cerr << max_rho << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (params->use_privacy_filter and params->approximate_privacy_filter) {
            rdp_accountant.setup_approximation();
        }
    }

    const int max_nb_trees = params->use_privacy_filter ? params->nb_trees + params->pf_additional_nb_trees : params->nb_trees + params->additional_nb_trees;
    if (!params->use_privacy_filter & params->additional_nb_trees > params->nb_trees) {
        std::cerr << "The number of additional trees for after regular training can only be as large as the number of trees from regular training to avoid further privacy leakage." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // train all trees
    for(int tree_index = 0; tree_index < max_nb_trees;  tree_index++) {

        update_splits(tree_index);

        // determine current tree positioning in ensemble
        if ((tree_index % params->nb_trees) == 0) {  // reset dataset for additional rounds due to Rényi filter
            *dataset = dataset_cpy;
            if (params->continuous_learning && tree_index == 0) {
                // Filter out subpopulation at beginning of training
                std::vector<int> remaining_indices;
                for (size_t i=0; i<dataset->length; i++) {
                    if (dataset_cpy.name.find("abalone") != std::string::npos) {
                        if (dataset->y[i] <= -0.36) {
                            remaining_indices.push_back(i);
                        }
                    } else if (dataset_cpy.name.find("adult") != std::string::npos) {
                        if (dataset->y[i] != 1) {
                            remaining_indices.push_back(i);
                        }
                    } else if (dataset_cpy.name.find("custom") != std::string::npos) {
                        if (dataset->cluster_ids[i] == 0) {
                            remaining_indices.push_back(i);
                        }
                    }
                }

                filtered_dataset = dataset->get_subset(remaining_indices);
                novel_dataset = dataset->remove_rows(remaining_indices);
                this->dataset = &filtered_dataset;

                // compute initial prediction
                this->init_score = params->task->compute_init_score(dataset->y, params->init_score_threshold, params->use_dp, params->privacy_budget * params->privacy_budget_init_score_ratio);
                LOG_DEBUG("Training initialized with score: {1}", init_score);
            }
        }

        if (params->continuous_learning && tree_index == params->nb_trees) {
            if (params->use_privacy_filter) {
                // Add back in subpopulation
                *dataset = dataset_cpy;
            } else {
                // Without Rényi filter, the ensemble must forget old data
                *dataset = novel_dataset;
                // Start accounting again for additional rounds after regular training 
                if (params->leaf_noise == GAUSS) {
                    Accounting accounting;
                    if (params->custom_noise_scale > 0.0) {
                        // Use custom noise scale
                        accounting = rdp_accountant.setup_accounting(params->additional_nb_trees, params->custom_noise_scale);
                        tree_params->active_noise_scale = params->custom_noise_scale;
                    } else {
                        // Finetune to optimal noise scale
                        accounting = rdp_accountant.setup_accounting(params->additional_nb_trees);
                    }
                    tree_params->rdp_alpha = accounting.alpha;
                    tree_params->active_noise_scale = accounting.noise_scale;

                    if (accounting.delta > DELTA) {
                        std::cerr << "Accounted delta is too large delta=" << accounting.delta << "." << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    if (max_rho < 0.0) {
                        std::cerr << "Failure in accounting: max_rho/apx_n_max_rho is negative." << std::endl;
                        std::cerr << max_rho << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }

            }
        }
  
        update_gradients(&dataset_cpy, dataset_cpy.gradients, tree_index, true);
        dataset->gradients = std::vector<double>(dataset->length, 0.0);
        if (params->newton_boosting) {
            update_hessians(&dataset_cpy, dataset_cpy.hessians);
            dataset->hessians = std::vector<double>(dataset->length, 0.0);
            for (int i=0; i<dataset->length; i++) {
                dataset->gradients[i] = dataset_cpy.gradients[dataset->identifiers[i]];
                dataset->hessians[i] = dataset_cpy.hessians[dataset->identifiers[i]];
            }
        } else {
            for (int i=0; i<dataset->length; i++) {
                dataset->gradients[i] = dataset_cpy.gradients[dataset->identifiers[i]];
            }
        }

        if(params->use_dp) {   // build a dp-tree

            vector<int> tree_indices;
            bool gradient_filtering = params->gradient_filtering && params->leaf_noise == LAPLACE;

            if (params->use_privacy_filter) {
                // Allow for different l2_threshold in additional iterations
                if (tree_index == params->nb_trees) {
                    tree_params->active_threshold = params->pf_l2_threshold;
                    tree_params->hess_active_threshold = params->pf_hess_l2_threshold;
                    tree_params->active_subsampling_ratio = params->subsampling_ratio * params->pf_subsampling_ratio_factor;
                    if (params->approximate_privacy_filter) {
                        // Refill the approximation table with new parameters.
                        rdp_accountant.setup_approximation();
                    }
                }
                for (int i=0; i<dataset->length; i++) {
                    double clipped_gradient = clamp(dataset->gradients[i], -tree_params->active_threshold, tree_params->active_threshold);
                    if (tree_index >= params->nb_trees) {
                        // additional iterations
                        // clip gradient on active dataset
                        dataset->gradients[i] = clipped_gradient;       
                    }
                    
                    double ind_sens = rdp_accountant.gen_individual_sens(std::fabs(clipped_gradient));
                    double hess_ind_sens = params->newton_boosting ? dataset->hessians[i] : 1.0;
                    double ind_rho = params->approximate_privacy_filter ? rdp_accountant.approximate_rho(ind_sens, hess_ind_sens) : rdp_accountant.gen_rho(ind_sens, hess_ind_sens);                    

                    if (accounted_rho[dataset->identifiers[i]] + ind_rho <= max_rho + 10e-7) {
                        if (tree_index >= params->nb_trees) {
                            // additional iterations
                            // keep datapoints that have remaining budget
                            tree_indices.push_back(i);
                        }
                        // update privacy filter
                        accounted_rho[dataset->identifiers[i]] += ind_rho;
                    } else {
                        if (tree_index < params->nb_trees) {
                            // regular iterations
                            accounted_rho[dataset->identifiers[i]] = max_rho;
                        }
                    }
                    
                    if (accounted_rho[dataset->identifiers[i]] > max_rho + 10e-5) {
                        std::cout << "overstepped budget" << std::endl;
                        std::cout << "datapoint " << i << ": " << accounted_rho[i] << " + " << ind_rho << " / " << max_rho << std::endl;
                        std::exit(EXIT_FAILURE);
                    } 
                }
            }

            // sensitivity for internal nodes
            switch (params->criterion) {
                case XGD_MSE:
                    // This sensitivity is valid for unbounded DP.
                    tree_params->delta_g = 3 * pow(tree_params->active_threshold, 2);
                    break;
                case XGBOOST:
                    if (this->tree_params->gain_privacy_share > 0.)
                        std::cout << "Warning: split gain has no sensitivity." << std::endl;
                    break;
            }

            // if active, apply privacy filter once we reached additional iterations
            if (params->use_privacy_filter && tree_index >= params->nb_trees) {
                LOG_INFO("PrivFilt: {1} of {2} rows have leftover budget",
                        tree_indices.size(), dataset->length);

                if (tree_indices.size() == 0) {
                    return;
                }

                // PF is active, so we are using gauss: Generate a poisson subsample
                // Sample the number of rows, then sample the datapoints
                std::binomial_distribution<> d(tree_indices.size(), tree_params->active_subsampling_ratio);
                int number_of_rows = d(gen);
                // If the subsample is empty, the tree will be trained without any data points.
                random_unique(tree_indices, number_of_rows);
                
                LOG_INFO("PrivFilt: {1} of {2} rows will be used",
                    tree_indices.size(), dataset->length);
            }
            else { // regular iteration
                // determine number of rows
                int number_of_rows;
                
                if (params->leaf_noise == GAUSS) {
                    // When using gauss, generate a poisson subsample
                    // Sample the number of rows, then sample the datapoints
                    std::binomial_distribution<> d(dataset->length, tree_params->active_subsampling_ratio);
                    number_of_rows = d(gen);
                } else {
                    if (params->balance_partition) {
                        // num_unused_rows / num_remaining_trees
                        number_of_rows = dataset->length / (params->nb_trees - tree_index);
                    } else {
                        // line 8 of Algorithm 2 from DPBoost paper
                        number_of_rows = (original_length * params->learning_rate *
                                std::pow(1 - params->learning_rate, tree_index)) / 
                                (1 - std::pow(1 - params->learning_rate, params->nb_trees));
                        if (number_of_rows == 0) {
                            std::cout << "warning, tree with no samples left" << std::endl;
                            return;
                        }
                    }
                }

                // gradient-based data filtering
                if(gradient_filtering) {
                    std::vector<int> reject_indices, remaining_indices;
                    for (int i=0; i<dataset->length; i++) {
                        double curr_grad = dataset->gradients[i];
                        if (curr_grad < -tree_params->active_threshold or curr_grad > tree_params->active_threshold) {
                            reject_indices.push_back(i);
                        } else {
                            remaining_indices.push_back(i);
                        }
                    }
                    LOG_INFO("GDF: {1} of {2} rows fulfill gradient criterion",
                        remaining_indices.size(), dataset->length);

                    tree_indices = remaining_indices;

                    if (static_cast<size_t>(number_of_rows) <= remaining_indices.size()) {
                        // we have enough samples that were not filtered out
                        random_unique(tree_indices, number_of_rows);  // take n_rows which were shuffled before 
                    } else {
                        // suboptimal but acc. to Li et al. we have to deactivate using out-filtered grads via clipping
                        if (false) {
                            // we don't have enough -> take all samples that were not filtered out
                            // and fill up with randomly chosen and clipped filtered ones

                            LOG_INFO("GDF: filling up with {1} rows (clipping those gradients)",
                                number_of_rows - tree_indices.size());
                            std::random_shuffle(reject_indices.begin(), reject_indices.end());

                            int reject_index = 0;
                            for(size_t i=tree_indices.size(); i<static_cast<size_t>(number_of_rows); i++){
                                int curr_index = reject_indices[reject_index++];
                                dataset->gradients[curr_index] = clamp(dataset->gradients[curr_index],
                                    -params->l2_threshold, params->l2_threshold);
                                tree_indices.push_back(curr_index);
                            }
                        } else {
                            if (remaining_indices.empty()) {
                                LOG_INFO("GDF removed all samples in tree {1}; mocking empty tree.", tree_index);
                            }
                        }
                    }
                } else {
                    // no GDF, just randomly select <number_of_rows> rows.
                    if (params->leaf_noise != GAUSS or tree_params->active_subsampling_ratio < 1.0) {
                        tree_indices = vector<int>(dataset->length);
                        std::iota(std::begin(tree_indices), std::end(tree_indices), 0);
                        random_unique(tree_indices, number_of_rows);
                    }
                }
            }

            DataSet tree_dataset;
            if (params->leaf_noise == GAUSS and tree_params->active_subsampling_ratio == 1.0) {
                tree_dataset = *dataset;
            } else {
                tree_dataset = dataset->get_subset(tree_indices);
            }

            if (!gradient_filtering && (!params->use_privacy_filter || tree_index < params->nb_trees)) {
                // no GDF or still in regular PF rounds, so we clip the gradients to bound the sensitivity for splits and leaves
                for (int i=0; i<tree_dataset.length; i++) {
                    tree_dataset.gradients[i] = clamp(tree_dataset.gradients[i], -tree_params->active_threshold, tree_params->active_threshold);
                }
            }
            
            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"), tree_index, tree_params->tree_privacy_budget, tree_dataset.length);

            // build tree
            LOG_INFO("Building dp-tree-{1} using {2} samples...", tree_index, tree_dataset.length);
            DPTree tree = DPTree(params, tree_params, &tree_dataset, tree_index);
            tree.fit();
            trees.push_back(tree);

            if (params->leaf_noise == LAPLACE) {
                // remove rows
                *dataset = dataset->remove_rows(tree_indices);
            }

        } else {  // build a non-dp tree
            
            LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"), tree_index, tree_params->tree_privacy_budget, dataset->length);

            // build tree
            LOG_INFO("Building non-dp-tree {1} using {2} samples...", tree_index, dataset->length);
            DPTree tree = DPTree(params, tree_params, dataset, tree_index);
            tree.fit();
            trees.push_back(tree);
        }

        // print the tree if we are in debug mode
        if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
            trees.back().recursive_print_tree(trees.back().root_node);
        }
        LOG_INFO(YELLOW("Tree {1:2d} done. Instances left: {2}"), tree_index, dataset->length);
    }
}


vector<double> DPEnsemble::predict(VVD &X) const
{
    return predict(X, trees.size());
}


// Predict values from the ensemble of gradient boosted trees
vector<double> DPEnsemble::predict(VVD &X, size_t treestop_early) const
{
    vector<double> predictions(X.size(),0);
    vector<double> single_tree_pred(X.size());  // reuse single tree predictions
    size_t i = 0;
    for (auto &tree : trees) {
        if (i++ >= treestop_early) break;
        tree.predict(X, single_tree_pred);
        std::transform(single_tree_pred.begin(), single_tree_pred.end(), 
            predictions.begin(), predictions.begin(), std::plus<double>());
    }

    const double innit_score = this->init_score;
    const double learning_rate = params->learning_rate;
    std::transform(predictions.begin(), predictions.end(), predictions.begin(), 
            [learning_rate, innit_score](double &c){return c*learning_rate + innit_score;});

    return predictions;
}

// Predict values from the ensemble of gradient boosted trees
vector<double> DPEnsemble::predict_without_lr_init(VVD &X) const
{
    vector<double> predictions(X.size(),0);
    vector<double> single_tree_pred(X.size());  // reuse single tree predictions
    for (auto &tree : trees) {
        tree.predict(X, single_tree_pred);
        std::transform(single_tree_pred.begin(), single_tree_pred.end(), 
            predictions.begin(), predictions.begin(), std::plus<double>());
    }
    return predictions;
}

std::vector<double> DPEnsemble::predict_cached(VVD &X) const {

    vector<double> predictions(X.size(),0);
    vector<double> single_tree_pred(X.size());

    if (!trees.empty()) {
        trees.back().predict(X, single_tree_pred);
        std::transform(single_tree_pred.begin(), single_tree_pred.end(), 
            tree_params->saved_predictions.begin(), tree_params->saved_predictions.begin(), std::plus<double>());
    }

    const double innit_score = this->init_score;
    const double learning_rate = params->learning_rate;
    std::transform(tree_params->saved_predictions.begin(), tree_params->saved_predictions.end(), predictions.begin(), 
            [learning_rate, innit_score](double &c){return c*learning_rate + innit_score;});

    return predictions;
}

void DPEnsemble::getTrees(VVI &split_attr, VVD &split_val,
                          std::vector<VVI> &next_node, VVD &leafs) {
    if (!split_attr.empty() || !split_val.empty() || !next_node.empty()  || !leafs.empty()) return;
    split_attr.reserve(this->trees.size());
    split_val.reserve(this->trees.size());
    next_node.reserve(this->trees.size());
    leafs.reserve(this->trees.size());
    int max_tree_depth = this->params->max_depth;

    for (auto &t : this->trees) {
        std::vector<double> split_val_, leafs_;
        std::vector<int> split_attr_;
        VVI next_node_;
        split_val_.reserve(pow(2, max_tree_depth));
        leafs_.reserve(pow(2, max_tree_depth));
        split_attr_.reserve(pow(2, max_tree_depth));
        split_val_.reserve(pow(2, max_tree_depth));

        DPTree::getTree(t.root_node, split_attr_, split_val_, next_node_, leafs_);
        split_attr.push_back(split_attr_);
        split_val.push_back(split_val_);
        next_node.push_back(next_node_);
        leafs.push_back(leafs_);
    };
    
}


void DPEnsemble::exportTrees(VVD &value, VVI &children_left, VVI &children_right,
                VVD &threshold, std::vector<std::vector<std::complex<double>>> &impurity, VVI &feature, VVI &n_node_samples)
{
    if (!value.empty() || !children_left.empty() || !children_right.empty()  || !threshold.empty()
        || !impurity.empty() || !feature.empty() || !n_node_samples.empty()) return;
    value.reserve(this->trees.size());
    children_left.reserve(this->trees.size());
    children_right.reserve(this->trees.size());
    threshold.reserve(this->trees.size());
    impurity.reserve(this->trees.size());
    feature.reserve(this->trees.size());
    n_node_samples.reserve(this->trees.size());
    int max_tree_depth = this->params->max_depth;

    for (auto &t : this->trees) {
        std::vector<double> value_, threshold_;
        std::vector<std::complex<double>> impurity_;
        std::vector<int> children_left_, children_right_, feature_, n_node_samples_;
        value_.reserve(pow(2, max_tree_depth));
        children_left_.reserve(pow(2, max_tree_depth));
        children_right_.reserve(pow(2, max_tree_depth));
        threshold_.reserve(pow(2, max_tree_depth));
        impurity_.reserve(pow(2, max_tree_depth));
        feature_.reserve(pow(2, max_tree_depth));
        n_node_samples_.reserve(pow(2, max_tree_depth));

        DPTree::exportTree(t.root_node, value_, children_left_, children_right_,
            threshold_, 0, params->l2_lambda, impurity_, feature_, n_node_samples_);
        value.push_back(value_);
        children_left.push_back(children_left_);
        children_right.push_back(children_right_);
        threshold.push_back(threshold_);
        impurity.push_back(impurity_);
        feature.push_back(feature_);
        n_node_samples.push_back(n_node_samples_);
    };
    
}


void DPEnsemble::update_gradients(DataSet *dataset, std::vector<double> &gradients, int tree_index, bool cache_prediction)
{
    if(tree_index == 0) {
        // init gradients
        vector<double> init_scores(dataset->length, init_score);
        gradients = params->task->compute_gradients(dataset->y, init_scores, params->newton_boosting);
    } else { 
        // update gradients
        vector<double> y_pred;
        if (cache_prediction) {
             y_pred = predict_cached(dataset->X);
        } else {
            y_pred = predict(dataset->X);
        }
        gradients = (params->task)->compute_gradients(dataset->y, y_pred, params->newton_boosting);
    }
}

void DPEnsemble::update_hessians(DataSet *dataset, std::vector<double> &hessians)
{
    hessians = params->task->compute_hessians(dataset->y, dataset->gradients);
}

void DPEnsemble::update_splits(int tree_index) {
    if (tree_index == 0) {
        for (int feature=0; feature<this->dataset->num_x_cols; feature++) {
            double start = params->feature_val_border.first;
            double end = params->feature_val_border.second;
            double step = (end - start) / params->num_split_candidates;
            std::vector<double> new_vec;
            tree_params->split_candidates.push_back(new_vec);
            for (int i=0; i<params->num_split_candidates; i++) {
                tree_params->split_candidates[feature].push_back(start + i * step);
            }
        }
    } else {

        if (!params->refine_splits or tree_index >= params->refine_splits_rounds) return;

        int feature = tree_index % dataset->num_x_cols;

        double total_hess = 0.0;
        std::vector<double> hess_hist(tree_params->split_candidates[feature].size() + 1);
    
        std::vector<int> indices(dataset->length);
        std::iota(indices.begin(), indices.end(), 0);
        // Generate poisson subsample
        if (params->refine_splits_subsample < 1.0) {
            std::binomial_distribution<> d(indices.size(), params->refine_splits_subsample);
            std::random_device rd;
            std::mt19937 gen(rd());
            int number_of_rows = d(gen);
            random_unique(indices, number_of_rows);
        }

        for (int index : indices) {
            for (size_t i=0; i<hess_hist.size(); i++) {
                const double clipped_hessian = params->newton_boosting 
                    ? clamp(dataset->hessians[index], -params->hess_l2_threshold, params->hess_l2_threshold) 
                    : 1.0;
                if (i == 0 and dataset->X[index][feature] < tree_params->split_candidates[feature][i]) {
                    hess_hist[i] += clipped_hessian;
                } else if (i > 0 and i < hess_hist.size() - 1 
                and dataset->X[index][feature] >= tree_params->split_candidates[feature][i-1] && dataset->X[index][feature] < tree_params->split_candidates[feature][i]) {
                    hess_hist[i] += clipped_hessian;
                } else if (i == hess_hist.size() - 1
                and dataset->X[index][feature] >= tree_params->split_candidates[feature][i-1]) {
                    hess_hist[i] += clipped_hessian;
                }
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);
        const double sens = params->newton_boosting ? tree_params->hess_active_threshold : 1.0;
        for (size_t i=0; i<hess_hist.size(); i++) {
            hess_hist[i] += d(gen) * tree_params->active_noise_scale * sens * 1. / std::sqrt(params->leaf_denom_noise_weight * 2);
            total_hess += hess_hist[i];
        }

        std::vector<double> current_splits = tree_params->split_candidates[feature];
        std::vector<double> new_splits;
        double opt_freq = total_hess / params->num_split_candidates; // Uniform hess bins...

        std::replace_if(hess_hist.begin(), hess_hist.end(), [](double x) { return x < 0; }, 0.0);

        double total_freq = 0;
        for (size_t i = 0; i < hess_hist.size(); ++i) {
            total_freq += hess_hist[i];
            if (total_freq > opt_freq && i < current_splits.size() - 1) {
                new_splits.emplace_back(current_splits[i]);
                total_freq = 0;
                if (static_cast<int>(current_splits.size()) < params->num_split_candidates && static_cast<int>(new_splits.size()) < params->num_split_candidates) {
                    new_splits.emplace_back((current_splits[i] + current_splits[i + 1]) / 2);
                }
            } else if (i == (current_splits.size() - 1)) {
                new_splits.emplace_back(current_splits[i]);
            }
        }

        std::sort(new_splits.begin(), new_splits.end());
        std::unique(new_splits.begin(), new_splits.end());

        if (static_cast<int>(new_splits.size()) < params->num_split_candidates) {
            int factor = std::ceil(static_cast<double>(params->num_split_candidates) / new_splits.size());
            if (factor >= 1) {
                std::vector<double> temp_splits;
                for (size_t i = 0; i < new_splits.size() - 1; ++i) {
                    double start = new_splits[i];
                    double end = new_splits[i + 1];
                    double step = (end - start) / factor;
                    for (int j = 0; j < factor; ++j) {
                        temp_splits.push_back(start + j * step);
                    }
                }
                new_splits.insert(new_splits.end(), temp_splits.begin(), temp_splits.end());
            }
        }

        // Sort and remove duplicates
        std::sort(new_splits.begin(), new_splits.end());
        std::vector<double>::iterator new_end = std::unique(new_splits.begin(), new_splits.end());
        new_splits.resize(std::distance(new_splits.begin(), new_end));
        tree_params->split_candidates[feature] = new_splits;
    }

    if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
        int feature = tree_index % dataset->num_x_cols;
        std::cout << "Feature " << feature << std::endl;
        std::cout << tree_params->split_candidates[feature].size() << " candidates" << std::endl;
        for (size_t i=0; i<tree_params->split_candidates[feature].size(); i++) {
            std::cout << tree_params->split_candidates[feature][i] << ", ";
        }
        std::cout << std::endl;
    }
    
    
}
