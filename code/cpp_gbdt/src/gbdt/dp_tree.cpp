#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <tuple>
#include "dp_tree.h"

extern bool AVGLEAKAGE_MODE;
extern std::once_flag flag_featbord;
extern std::ofstream avgleakage;

using namespace std;


/** Constructors */

DPTree::DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index): 
    params(params),
    tree_params(tree_params), 
    dataset(dataset),
    tree_index(tree_index) {
    
        // prepare select features
        switch(params->max_features) {
            case RAND :
            case ONE : this->max_features = static_cast<size_t>( 1 ); break;
            case LOG2 : this->max_features = static_cast<size_t>( log2(dataset->num_x_cols) ); break;
            case SQRT : this->max_features = static_cast<size_t>( sqrt(dataset->num_x_cols) ); break;
            case ALL :
            default : this->max_features = static_cast<size_t>( dataset->num_x_cols ); break;
        }
        // ensure we always select one feature at minimum
        this->max_features = std::max(this->max_features, static_cast<size_t>( 1 ));
    }

DPTree::~DPTree() {}


/** Methods */

// Fit the tree to the data
void DPTree::fit()
{
    // keep track which samples will be available in a node for spliting
    vector<int> live_samples(dataset->length);
    std::iota(std::begin(live_samples), std::end(live_samples), 0);

    // keep track which features will be available for each tree depth
    vector<int> live_attr(dataset->num_x_cols);
    std::iota(std::begin(live_attr), std::end(live_attr), 0);

    int feature_index = tree_index % dataset->num_x_cols;
    this->root_node = make_tree_DFS(0, live_samples, live_attr, std::pair<int, int>{0, tree_params->split_candidates[feature_index].size() - 1});

    if(params->use_dp) {

        // add laplace/gauss noise to leaf values
        add_leaf_noise((1.0 - tree_params->gain_privacy_share) * tree_params->tree_privacy_budget);
    }
}


// Recursively build tree, DFS approach, first instance returns root node
TreeNode *DPTree::make_tree_DFS(int current_depth, vector<int> live_samples, vector<int> live_attr, std::pair<int, int> active_split_range)
{
    // max depth reached or not enough samples or split range too small -> leaf node 
    if (current_depth == params->max_depth 
    || live_samples.size() < static_cast<size_t>(params->min_samples_split) 
    || (!params->ignore_split_constraints and (active_split_range.first >= active_split_range.second))) {
        TreeNode *leaf = make_leaf_node(current_depth, live_samples);
        LOG_DEBUG("max_depth ({1}) or min_samples ({2})-> leaf (pred={3:.2f})", current_depth, live_samples.size(), leaf->prediction);
        return leaf;
    }

    // get the samples (and their gradients) that actually end up in this node
    // note that the cols of X are rows in X_live
    VVD X_live;
    X_live.reserve(dataset->num_x_cols);
    for(int col=0; col < dataset->num_x_cols; col++) {
        vector<double> temp( live_samples.size() );
        for (size_t row = 0; row < live_samples.size(); ++row) {
            temp[row] = (dataset->X)[live_samples[row]][col];
        }
        X_live.push_back(temp);
    }
    vector<double> gradients_live( live_samples.size() );
    for (size_t row = 0; row < live_samples.size(); ++row) {
        gradients_live[row] = (dataset->gradients)[live_samples[row]];
    }

    // find best split
    double sum_grads_live = std::accumulate(gradients_live.begin(), gradients_live.end(), 0.0);
    TreeNode *node = find_best_split(X_live, gradients_live, sum_grads_live, live_attr, current_depth, active_split_range);

    // no split found
    if (node->is_leaf()) {
        LOG_DEBUG("no split found -> leaf");
        delete node;
        return make_leaf_node(current_depth, live_samples);
    }

    LOG_DEBUG("best split @ {1}, val {2:.2f}, gain {3:.5f}, curr_depth {4}, samples {5} ->({6},{7})", node->split_attr, node->split_value, node->split_gain, current_depth, node->lhs_size + node->rhs_size, node->lhs_size, node->rhs_size);

    // prepare the new live samples to continue recursion
    vector<char> lhs;
    bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(),
            node->split_attr) != (params->cat_idx).end();
    (void) samples_left_right_partition(lhs, X_live, node->split_attr, node->split_value, categorical);
    vector<int> left_live_samples, right_live_samples;
    for (size_t i=0; i<live_samples.size(); i++) {
        if (lhs[i]) {
            left_live_samples.push_back(live_samples[i]);
        } else {
            right_live_samples.push_back(live_samples[i]);
        }
    }

    vector<int> attr_live = live_attr;
    if (!params->reuse_attr) {
        attr_live.erase(std::remove(attr_live.begin(), attr_live.end(), node->split_attr), attr_live.end());
    }
    node->left = make_tree_DFS(current_depth + 1, left_live_samples, attr_live, std::pair<double, double>{active_split_range.first, node->split_index - 1});
    node->right = make_tree_DFS(current_depth + 1, right_live_samples, attr_live, std::pair<double, double>{node->split_index + 1, active_split_range.second});

    return node;
}


TreeNode *DPTree::make_leaf_node(int current_depth, vector<int> &live_samples)
{
    TreeNode *leaf = new TreeNode(true);
    leaf->depth = current_depth;
    leaf->n = live_samples.size();

    if (!live_samples.empty()) {
        leaf->identifiers = std::vector<int>( live_samples.size() );
        for (size_t i = 0; i < live_samples.size(); i++) {
            leaf->identifiers[i] = live_samples[i];
        }
    }

    if (leaf->n == 0) {  // e.g. because of random trees
        leaf->prediction = 0;
    } else {
        // in the latter case we will need this prediction later on
        if (!params->use_dp || params->leaf_denom_noise_weight == 0.0) { 
            vector<double> gradients;
            vector<double> hessians;
            for (auto index : live_samples) {
                gradients.push_back(dataset->gradients[index]);
                if (params->newton_boosting) {
                    hessians.push_back(dataset->hessians[index]);
                }
            }
            const double nominator = std::accumulate(gradients.begin(), gradients.end(), 0.0);
            const double denominator = params->newton_boosting ? std::accumulate(hessians.begin(), hessians.end(), 0.0) 
            : static_cast<double>( gradients.size() );
            // compute prediction
            double reg_denominator = 0.0;
            if (params->lambda_reg_mode == MAX) {
                reg_denominator = std::max(denominator, params->l2_lambda);
            } else if (params->lambda_reg_mode == ADD) {
                reg_denominator = denominator + params->l2_lambda;
            }
            leaf->prediction = (-1 * nominator / reg_denominator);
            if (params->reg_delta != 0.0) {
                leaf->prediction = clamp(leaf->prediction, -params->reg_delta, params->reg_delta);
            }
        }
    }
    leaves.push_back(leaf);
    return leaf;
}


void DPTree::predict(VVD &X, std::vector<double> &predictions) const
{
    // save computations when 'prediction' is reused, else pre-init vector here
    if (predictions.size() != X.size()) {
        predictions = vector<double>( X.size() );
    }

    // iterate over all samples
    std::transform(X.cbegin(), X.cend(), predictions.begin(), [this](vector<double> row) {
        return _predict(&row, this->root_node);
    });
}


// recursively walk through decision tree
double DPTree::_predict(vector<double> *row, TreeNode *node) const
{
    if(node->is_leaf()){
        return node->prediction;
    }
    double row_val = (*row)[node->split_attr];

    if (std::find((params->cat_idx).begin(), (params->cat_idx).end(), node->split_attr) != (params->cat_idx).end()) {
        // categorical feature
        if (row_val == node->split_value){
            return _predict(row, node->left);
        }
    } else { // numerical feature
        if (row_val <= node->split_value){
            return _predict(row, node->left);
        }
    }
    return _predict(row, node->right);
}


// find best split of data using the exponential mechanism
TreeNode *DPTree::find_best_split(VVD &X_live, vector<double> &gradients_live, double sum_grads_live,
                vector<int> &attr_live, int current_depth, std::pair<int, int> active_split_range)
{
    double privacy_budget_for_node = tree_params->gain_privacy_share * tree_params->tree_privacy_budget / params->max_depth;

    vector<SplitCandidate> probabilities;
    int size = static_cast<int>(gradients_live.size());

    // select features based on feature weights for random feature_values strategy
    std::vector<int> selected_features;

    if (params->cyclical_feature_interactions) {
        const int num_features = (params->cat_idx.size() + params->num_idx.size());
        selected_features.push_back(this->tree_index % num_features);
    } else {
        if (params->max_feature_values == RAND) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::vector<double> feature_weight = tree_params->feature_weights;
            selected_features = std::vector<int>(this->max_features);

            for (size_t i = 0; i < selected_features.size(); ++i) {
                std::discrete_distribution<> d(feature_weight.cbegin(), feature_weight.cend());
                selected_features[i] = d(gen);
                feature_weight.erase(feature_weight.begin() + selected_features[i]);
            }
        } else {
            selected_features = attr_live;
            random_unique(selected_features, this->max_features);
        }
    }

    int split_index = -1;
    TreeNode *node;

    // iterate over features
    for (auto feature_index : selected_features) {
        const bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), feature_index) != (params->cat_idx).end();

        // just take unique feature values
        std::vector<double> uniq_feat_vals;
        if (params->max_feature_values != RAND) {
            uniq_feat_vals = X_live[feature_index];
            std::sort( uniq_feat_vals.begin(), uniq_feat_vals.end() );
            uniq_feat_vals.erase( std::unique( uniq_feat_vals.begin(), uniq_feat_vals.end() ), uniq_feat_vals.end() );
        }

        // prepare number of to-be-selected features
        size_t max_feature_values;
        switch(params->max_feature_values) {
            case RAND :
            case ONE : max_feature_values = static_cast<size_t>( 1 ); break;
            case LOG2 : max_feature_values = static_cast<size_t>( log2(uniq_feat_vals.size()) ); break;
            case SQRT : max_feature_values = static_cast<size_t>( sqrt(uniq_feat_vals.size()) ); break;
            case ALL :
            default : max_feature_values = uniq_feat_vals.size(); break;
        }
        // ensure we always select one feature value at minimum
        max_feature_values = std::max(max_feature_values, static_cast<size_t>( 1 ));

        // select features
        // for size=0 we mock-up an empty splits
        if (params->max_feature_values == RAND || size == 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            if (categorical && !params->cat_values[feature_index].empty()) {
                std::uniform_int_distribution<> dis(
                    params->cat_values[feature_index].front(),
                    params->cat_values[feature_index].back()
                );
                uniq_feat_vals = std::vector<double> { static_cast<double>( dis(gen) ) };
            } else {
                if (size > 0 && params->feature_val_border.first == 0 && params->feature_val_border.second == 0) {
                    if (params->use_dp) {
                        std::call_once(flag_featbord, [](){std::cout << "!!! Invalid DP guarantees !!! (better set feature_val_border!={0,0})" << std::endl;});
                    }
                    std::uniform_real_distribution<> dis(
                        *std::min_element(uniq_feat_vals.cbegin(), uniq_feat_vals.cend()),
                        *std::max_element(uniq_feat_vals.cbegin(), uniq_feat_vals.cend())
                    );
                    uniq_feat_vals = std::vector<double> { dis(gen) };
                } else {
                    if (params->random_splits_from_candidates) {
                        std::vector<double> split_candidates_for_feature = tree_params->split_candidates[feature_index];
                        
                        int split_start = 0;
                        int split_end = split_candidates_for_feature.size() - 1;
                        if (!params->ignore_split_constraints) {
                            split_start = active_split_range.first;
                            split_end = active_split_range.second;
                        }

                        std::uniform_int_distribution<int> uniform_dist(split_start, split_end);
                        split_index = uniform_dist(gen);
                        uniq_feat_vals = std::vector<double> { split_candidates_for_feature[split_index] };
                    } else {
                        std::uniform_real_distribution<> dis(
                            params->feature_val_border.first,
                            params->feature_val_border.second
                        );
                        uniq_feat_vals = std::vector<double> { dis(gen) };
                    }
                }
            }
        } else {
            random_unique(uniq_feat_vals, max_feature_values);
            std::sort( uniq_feat_vals.begin(), uniq_feat_vals.end() );

            // insert feature_val_borders into uniq_feat_vals and cut outliers for privacy reasons
            if ( params->use_dp && !categorical && ( params->feature_val_border.first != 0 || params->feature_val_border.second != 0) ) {
                // delete everything below or equal to lower feature border
                std::vector<double>::const_iterator lb = std::upper_bound( uniq_feat_vals.cbegin(), uniq_feat_vals.cend(), params->feature_val_border.first );
                uniq_feat_vals.erase(uniq_feat_vals.cbegin(), lb);

                // delete everything above or equal to upper feature border
                std::vector<double>::const_iterator ub = std::lower_bound( uniq_feat_vals.cbegin(), uniq_feat_vals.cend(), params->feature_val_border.second );
                uniq_feat_vals.erase(ub, uniq_feat_vals.cend());

                uniq_feat_vals.insert(uniq_feat_vals.cbegin(), params->feature_val_border.first );
                uniq_feat_vals.push_back( params->feature_val_border.second );
            }
            if ( params->use_dp && !categorical && params->feature_val_border.first == 0 && params->feature_val_border.second == 0 ) {
                std::call_once(flag_featbord, [](){std::cout << "!!! Invalid DP guarantees !!! (better set feature_val_border!={0,0})" << std::endl;});
            }
        }

        // before we can run the optimized gain objective
        // we have to sort the split candidates as well as the feature values
        std::vector<double> X_live_sorted(gradients_live.size()), grads_live_sorted(gradients_live.size());
        if (max_feature_values != 1) {
            std::vector<std::pair<double,double>> X_grad_paired(gradients_live.size());
            // make a pair of feature values and gradient as both are linked together during sorting
            for (size_t i = 0; i < gradients_live.size(); ++i) X_grad_paired[i] = std::make_pair(X_live[feature_index][i], gradients_live[i] );
            std::sort( X_grad_paired.begin(), X_grad_paired.end(), [](const std::pair<double,double> &l, const std::pair<double,double> &r) {
                    return l.first < r.first;
            } );
            // seperate the feature value - gradient pair again
            for (size_t i = 0; i < gradients_live.size(); ++i) {
                X_live_sorted[i] = X_grad_paired[i].first;
                grads_live_sorted[i] = X_grad_paired[i].second;
            }
        }

        double sum_grads_lhs = 0.0;
        int lhs_size = 0, categorical_skip = 0;
        for (std::vector<double>::const_iterator it_feature_value = uniq_feat_vals.cbegin(); it_feature_value != uniq_feat_vals.cend(); std::advance(it_feature_value, 1)) {
            // compute gain
            // always split no matter if it is useful or not.
            double orig_gain;
            if (max_feature_values == 1) {
                orig_gain = compute_gain_legacy(X_live, gradients_live, feature_index, *it_feature_value, lhs_size, categorical);
            } else {
                orig_gain = compute_gain(X_live_sorted, grads_live_sorted, sum_grads_live, sum_grads_lhs, *it_feature_value, lhs_size, categorical_skip, categorical);
            }

            double gain = orig_gain;
            if (params->use_dp && privacy_budget_for_node > 0.) {  // if we do not use DP, we don't need to alter the gain
                gain = (privacy_budget_for_node * gain) / (2.0 * tree_params->delta_g);
            }

            double bucket_width = 1.0;
            // determine bucket width to scale probs with; categorical features have non-data-dependent split values and thus no bucket width
            if (!categorical && params->max_feature_values != RAND && size > 0) {
                // the last bucket does not exist
                if (std::next(it_feature_value) == uniq_feat_vals.cend()) {
                    bucket_width = 0.0;
                } else {
                    // determine normalized bucket width (normalize with (x_max - x_min)/numeric_feature_weight)
                    bucket_width = params->numeric_feature_weight * (*std::next(it_feature_value) - *it_feature_value) / (uniq_feat_vals.back() - uniq_feat_vals.front() );
                }
            }

            // if bucket_width=0 then the probability of selecting this elem is 0 as well -> no need to add this elem to the pool
            if (fpclassify(bucket_width) != FP_ZERO) {
                SplitCandidate candidate = SplitCandidate(
                    feature_index, *it_feature_value, gain, orig_gain, bucket_width, true, lhs_size, size - lhs_size
                );
                probabilities.push_back(candidate);
            }
        }
    }


    // choose a split using the exponential mechanism
    int index = -1, top_index = -1;
    if (probabilities.size() == 1) {
        index = 0; // if there is only one (random) choice because of gain_eps=0 or no-data-splits, take it.
        top_index = 0;
    } else {
        index = exponential_mechanism(probabilities);

        // return the top index; only for export Tree purposes
        top_index = exponential_mechanism(probabilities, true);
    }

    // construct the node
    if (index == -1) {
        node = new TreeNode(true);
        std::cerr << "Warning: tree is probably not trained fully." << std::endl;
    } else {
        // select split value uniformly within this bucket
        double split_val_shift = 0.0;
        const bool categorical = std::find((params->cat_idx).begin(), (params->cat_idx).end(), probabilities[index].feature_index) != (params->cat_idx).end();
        if (!categorical && params->max_feature_values != RAND && size > 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(
                std::numeric_limits<double>::epsilon() * (probabilities[index].split_value + probabilities[index].bucket_width),
                probabilities[index].bucket_width
            );
            split_val_shift = dis(gen);
        }

        node = new TreeNode(false);
        node->split_attr = probabilities[index].feature_index;
        node->split_value = probabilities[index].split_value + split_val_shift;
        node->split_index = split_index;
        node->split_gain = probabilities[index].gain;
        node->split_gain_orig = probabilities[index].maximize * probabilities[index].orig_gain;  // only for export Tree purposes
        node->split_gain_top = probabilities[top_index].maximize * probabilities[top_index].orig_gain;  // only for export Tree purposes
        node->lhs_size = probabilities[index].lhs_size;
        node->rhs_size = probabilities[index].rhs_size;
    }
    node->depth = current_depth;
    return node;
}


/*
    Computes the gain of a split (fast: optimized O(n log n) calulation)

               sum(elem : IL)^2  +  sum(elem : IR)^2
    G(IL,IR) = ----------------     ----------------
                |IL| + lambda        |IR| + lambda
*/
double DPTree::compute_gain(std::vector<double> &samples, std::vector<double> &gradients_live, double sum_grads_live, double &sum_grads_lhs,
                double feature_value, int &lhs_size, int &categorical_skip, bool categorical)
{

    int _lhs_size;
    double lhs_gain, total_gain = 0;
    if(categorical) {
        // calculate pointer to old_lhs_size: first element greater than or equal to `feature_value`
        std::vector<double>::const_iterator lb = std::lower_bound( std::next(samples.cbegin(), categorical_skip), samples.cend(), feature_value );
        int dist_begin_lb = std::distance(samples.cbegin(), lb);
        // calculate pointer to new_lhs_size: first element strictly greater than `feature_value`
        std::vector<double>::const_iterator ub = std::upper_bound( std::next(samples.cbegin(), dist_begin_lb), samples.cend(), feature_value );
        int dist_begin_ub = std::distance(samples.cbegin(), ub);
        // lhs_gain <- all gradients summed together which are between old_lhs_size and new_lhs_size
        lhs_gain = std::accumulate(std::next(gradients_live.cbegin(), dist_begin_lb),
                                   std::next(gradients_live.cbegin(), dist_begin_ub), 0.0);

        _lhs_size = std::distance(lb, ub); // lhs_size <- new_lhs_size - old_lhs_size
        categorical_skip += _lhs_size; // increase the safe skip
    } else {
        // calculate pointer to new_lhs_size: first element greater than or equal to `feature_value`
        std::vector<double>::const_iterator lb = std::lower_bound( std::next(samples.cbegin(), lhs_size), samples.cend(), feature_value );
        _lhs_size = std::distance(samples.cbegin(), lb); // lhs_size <- new_lhs_size

        // add to running lhs gradient everything between old_lhs_size and new_lhs_size
        sum_grads_lhs += std::accumulate(std::next(gradients_live.cbegin(), lhs_size), std::next(gradients_live.cbegin(), _lhs_size), 0.0);
        lhs_gain = sum_grads_lhs; // lhs_gain <- current running gradient
    }

    double rhs_gain = sum_grads_live - lhs_gain;
    int _rhs_size = samples.size() - _lhs_size;
    lhs_size = _lhs_size;

    switch (params->criterion) {
        case XGD_MSE: {
            lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
            rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);

            total_gain = lhs_gain + rhs_gain;
            break;
        }
        case XGBOOST: {
            lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
            rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);
            const double add_gain = std::pow(lhs_gain + rhs_gain,2) / (_lhs_size + _rhs_size + params->l2_lambda);

            total_gain = lhs_gain + rhs_gain - add_gain;
            break;
        }
    }

    return std::max(total_gain, 0.0);
}

double DPTree::compute_gain_legacy(VVD &samples, vector<double> &gradients_live,
                int feature_index, double feature_value, int &lhs_size, bool categorical)
{
    // partition into lhs / rhs
    vector<char> lhs;
    int _lhs_size = samples_left_right_partition(lhs, samples, feature_index, feature_value, categorical);
    int _rhs_size = lhs.size() - _lhs_size;
    lhs_size = _lhs_size;

    double lhs_gain = 0, rhs_gain = 0, total_gain = 0;
    for (size_t index=0; index<lhs.size(); index++) {
        lhs_gain += lhs[index] * (gradients_live)[index];
        rhs_gain += (not lhs[index]) * (gradients_live)[index];
    }

    switch (params->criterion) {
        case XGD_MSE: {
            lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
            rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);

            total_gain = lhs_gain + rhs_gain;
            break;
        }
        case XGBOOST: {
            lhs_gain = std::pow(lhs_gain,2) / (_lhs_size + params->l2_lambda);
            rhs_gain = std::pow(rhs_gain,2) / (_rhs_size + params->l2_lambda);
            const double add_gain = std::pow(lhs_gain + rhs_gain,2) / (_lhs_size + _rhs_size + params->l2_lambda);

            total_gain = lhs_gain + rhs_gain - add_gain;
            break;
        }
    }
    return std::max(total_gain, 0.0);
}

// the result is an int array that will indicate left/right resp. 0/1
int DPTree::samples_left_right_partition(vector<char> &lhs, VVD &samples,
            int feature_index, double feature_value, bool categorical)
{
    vector<double> &sampl = samples[feature_index];
    lhs = vector<char>( sampl.size() );
    int lhs_active = 0;

    // if the feature is categorical
    if(categorical) {
        for (size_t i = 0; i < sampl.size(); ++i) {
            char value = sampl[i] == feature_value;
            lhs[i] = value;
            lhs_active += value;
        }
    } else { // feature is numerical
        for (size_t i = 0; i < sampl.size(); ++i) {
            char value = sampl[i] < feature_value;
            lhs[i] = value;
            lhs_active += value;
        }
    }
    return lhs_active;
}


// Computes probabilities from the gains. (Larger gain -> larger probability to 
// be chosen for split). Then a cumulative distribution function is created from
// these probabilities. Then we can sample from it using a RNG.
// The function returns the index of the chosen split.
int DPTree::exponential_mechanism(std::vector<SplitCandidate> &probs, const bool disable_dp)
{
    // if no split has a positive gain, return. Node will become a leaf
    // this action is not DP since this gain value is unprotected / unnoised
    if (false) {
        int count = std::count_if(probs.begin(), probs.end(),
            [](SplitCandidate c){ return c.gain > 0; });
        if (count == 0) {
            return -1;
        }
    }

    // non-dp: deterministically choose the best split
    if (!params->use_dp or disable_dp) {
        std::vector<SplitCandidate>::const_iterator max_elem = std::max_element(probs.cbegin(), probs.cend(), [](const SplitCandidate &a, const SplitCandidate &b) {
            return a.orig_gain < b.orig_gain;
        });
        // return index of the max_elem
        return std::distance(probs.cbegin(), max_elem);
    }

    // calculate the probabilities from the gains
    // bucket_width * exp(gain) <==> exp(gain + ln(bucket_width))
    vector<double> gains(probs.size()), probabilities(probs.size()), partials(probs.size());
    std::transform(probs.cbegin(), probs.cend(), gains.begin(), [](const SplitCandidate p) {
        return p.maximize * p.gain + std::log(p.bucket_width);
    });
    const double lse = log_sum_exp(gains);
    std::transform(gains.cbegin(), gains.cend(), probabilities.begin(), [lse](const double g) {
        // if a gain is negative we simply put less emphasize than if it is zero
        return exp(g - lse);
    });

    // create a cumulative distribution function from the probabilities.
    // all values will be in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    double rand01 = ((double) std::rand() / (RAND_MAX));

    // try to find a candidate at least 10 times before giving up and making the node a leaf node
    for (int tries=0; tries<10; tries++) {
        std::vector<double>::const_iterator first_elem_larger = std::lower_bound(partials.cbegin(), partials.cend(), rand01);  // O(log n)
        if (first_elem_larger != partials.cend()) {
            return std::distance(partials.cbegin(), first_elem_larger);  // O(1)
        }
        rand01 = ((double) std::rand() / (RAND_MAX));
    }
    return -1;
}

void DPTree::add_leaf_noise(double eps_leaf)
{
    std::random_device rd;
    std::function<double(const double, const double, const double)> gen_noise;
    // choose correct noise from Laplace or Gauss distribution for the leaves
    switch (params->leaf_noise) {
        case GAUSS: {
            std::mt19937 gen(rd());
            const double scale = tree_params->active_noise_scale;
            // This function returns noise scaled by the precomputed noise scale, weighted with noise_weight.
            gen_noise = [&gen, scale](const double sens, const double /*eps*/, const double noise_weight) {
                std::normal_distribution<> d(0.0, 1.0);
                const double dim = fpclassify(1.0 - noise_weight) == FP_ZERO ? 1 : 2; // number of dimensions
                return d(gen) * scale * sens * 1. / std::sqrt(noise_weight * dim);
            };
            break;
        }
        case LAPLACE:
        default: {
            Laplace lap(rd());
            gen_noise = [&lap](const double sens, const double eps, const double /*noise_weight*/){ return lap.return_a_random_variable(sens/eps); };
            break;
        }
    }

    for (auto &leaf : leaves) {

        double noised_nominator = 0.0;
        if (params->leaf_denom_noise_weight > 0.0) {
            for (auto id : leaf->identifiers) {
                noised_nominator += clamp(dataset->gradients[id], -tree_params->active_threshold, tree_params->active_threshold);
            }
            noised_nominator += gen_noise(tree_params->active_threshold, eps_leaf, 1.0 - params->leaf_denom_noise_weight);
        }

        double noised_denominator = 0.0;
        if (dynamic_cast<BinaryClassification*>(params->task.get()) != nullptr and params->newton_boosting) {
            for (auto id : leaf->identifiers) {
                noised_denominator += clamp(dataset->hessians[id], 0, tree_params->hess_active_threshold);
            }
            noised_denominator += 0.5 * gen_noise(tree_params->hess_active_threshold, -1.0, params->leaf_denom_noise_weight);
        } else {
            noised_denominator = params->leaf_denom_noise_weight > 0
            ? leaf->n + gen_noise(1.0, -1.0, params->leaf_denom_noise_weight)  // n has sensitivity 1.0; noise is 0-centered
            : 0.0;  // default: no adaptive DP n
        }

        LOG_DEBUG("({1:.3f} / {2:.3f})", noised_nominator, noised_denominator);

        if (params->cut_off_leaf_denom and noised_denominator < 0.0) {
            leaf->prediction = 0.0;
            continue;
        }

        double reg_noised_denominator = 0.0;
        // it is post-processing to lower bound / regularize n by lambda.
        if (params->lambda_reg_mode == ADD ) {   
            reg_noised_denominator = noised_denominator + params->l2_lambda;  
        } else if (params->lambda_reg_mode == MAX) {
            reg_noised_denominator = std::max(noised_denominator, params->l2_lambda);
        } else {
            std::cerr << "WARNING: Unknown lambda regularization mode " << lrt_to_str(params->lambda_reg_mode) << "." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        double leaf_sens = tree_params->active_threshold;
        if (params->leaf_clipping) {
            const double new_threshold = tree_params->active_threshold * std::pow((1.0 - params->learning_rate), this->tree_index);
            leaf->prediction = clamp(leaf->prediction, -new_threshold, new_threshold);
            leaf_sens = new_threshold;
        }

        // add noise from noise distribution to leaves: combined gradient (nominator) and n (denominator) noise
        const double noise = gen_noise(leaf_sens, eps_leaf, 1.0 - params->leaf_denom_noise_weight) / reg_noised_denominator;
        LOG_DEBUG("Adding {1:} noise to leaves (Scale {2:.2f}, N {3:.1f})", nt_to_str(params->leaf_noise), leaf_sens / reg_noised_denominator / eps_leaf, noised_denominator);
        const double noised_leaf_prediction = params->leaf_denom_noise_weight > 0
            ? -noised_nominator / reg_noised_denominator
            : leaf->prediction + noise;

        if (AVGLEAKAGE_MODE) {
            for (auto id : leaf->identifiers) {
                double prediction_with = -noised_nominator;
                double prediction_without = prediction_with + dataset->gradients[id];
                double leaf_sigma = tree_params->active_noise_scale * 1. / std::sqrt((1.0 - params->leaf_denom_noise_weight) * 2);
                double kl = std::pow(prediction_with - prediction_without, 2) / ( 2 *  std::pow(leaf_sigma, 2));
                avgleakage << dataset->identifiers[id] << "," << kl << std::endl;
            }
        }

        if (params->reg_delta != 0.0) {
            leaf->prediction = clamp(noised_leaf_prediction, -params->reg_delta, params->reg_delta);
        } else {
            leaf->prediction = noised_leaf_prediction;
        }
    }
}

void DPTree::getTree(TreeNode* node, std::vector<int> &split_attr, std::vector<double> &split_val,
                std::vector<std::vector<int>> &next_node, std::vector<double> &leafs)
{
    if (node->is_leaf()) {
        leafs.push_back(node->prediction);
        std::vector<int> next_{-1, -1, static_cast<int>(leafs.size()) - 1};
        next_node.push_back(next_);
        split_attr.push_back(-1);
        split_val.push_back(-1);
        return;
    }

    getTree(node->left, split_attr, split_val, next_node, leafs);
    size_t idx_left_child = next_node.size() - 1;

    getTree(node->right, split_attr, split_val, next_node, leafs);
    size_t idx_right_child = next_node.size() - 1;

    std::vector<int> next_{static_cast<int>(idx_left_child), static_cast<int>(idx_right_child), -1};
    split_attr.push_back(node->split_attr);
    split_val.push_back(node->split_value);
    next_node.push_back(next_);
}

void DPTree::exportTree(TreeNode* node, std::vector<double> &value, std::vector<int> &children_left,
                std::vector<int> &children_right, std::vector<double> &threshold, int node_id, double l2_lambda,
                std::vector<std::complex<double>> &impurity, std::vector<int> &feature, std::vector<int> &n_node_samples)
{
    if (node->is_leaf()) {
        value.push_back(node->prediction);
        threshold.push_back(-1);
        children_left.push_back(-1);
        children_right.push_back(-1);
        impurity.push_back(NAN);
        feature.push_back(-1);
        n_node_samples.push_back(node->n);
        return;
    }

    value.push_back(-1);  // filled later
    threshold.push_back(node->split_value);
    children_left.push_back(node_id + 1);
    children_right.push_back(-1);  // filled later
    impurity.push_back(std::complex<double>(node->split_gain_orig, node->split_gain_top));
    feature.push_back(node->split_attr);
    n_node_samples.push_back(node->lhs_size + node->rhs_size);

    exportTree(node->left, value, children_left, children_right, threshold,
                node_id + 1, l2_lambda, impurity, feature, n_node_samples);

    children_right[node_id] = value.size();

    exportTree(node->right, value, children_left, children_right, threshold,
                value.size(), l2_lambda, impurity, feature, n_node_samples);

    value[node_id] = (
        (
            value[children_left[node_id]] * (n_node_samples[children_left[node_id]] + l2_lambda)
            + value[children_right[node_id]] * (n_node_samples[children_right[node_id]] + l2_lambda)
        ) / (
            n_node_samples[children_left[node_id]] + n_node_samples[children_right[node_id]] + l2_lambda
        )
    );
}


// active in debug mode, prints the tree to console
void DPTree::recursive_print_tree(TreeNode* node) {

    if (node->is_leaf()) {
        return;
    }
    // check if split uses categorical attr
    bool categorical = std::find( params->cat_idx.begin(), params->cat_idx.end(), node->split_attr) != params->cat_idx.end();
    
    if (categorical) {
        std::cout << std::defaultfloat;
    } else {
        std::cout << std::fixed;
    }

    for (int i = 0; i < node->depth; ++i) { cout << ":  "; }

    if (!categorical) {
        cout << "Attr" << std::setprecision(3) << node->split_attr << 
            " < " << std::setprecision(3) << node->split_value << " (" << node->split_index <<")";
    } else {
        double split_value = (node->split_value); // categorical, hacked
        cout << "Attr" << node->split_attr << " = " << split_value;
    }
    if (node->left->is_leaf()) {
        cout << " (" << "L-leaf" << ") " << node->left->prediction << " (" << node->left->n << ")" << endl;
    } else {
        cout << endl;
    }

    recursive_print_tree(node->left);

    if (categorical) {
        std::cout << std::defaultfloat;
    } else {
        std::cout << std::fixed;
    }

    for (int i = 0; i < node->depth; ++i) { cout << ":  "; }
    if (!categorical) {
        cout << "Attr" << std::setprecision(3) << node->split_attr <<
            " >= " << std::setprecision(3) << node->split_value  << " (" << node->split_index <<")";
    } else {
        double split_value = node->split_value;
        cout << "Attr" << node->split_attr << " != " << split_value;
    }
    if (node->right->is_leaf()) {
        cout << " (" << "R-leaf" << ") " << node->right->prediction << " (" << node->right->n << ")" << endl;
    } else {
        cout << endl;
    }
    recursive_print_tree(node->right);
}


// free allocated ressources
void DPTree::delete_tree(TreeNode *node)
{
    if (not node->is_leaf()) {
        delete_tree(node->left);
        delete_tree(node->right);
    }
    delete node;
    return;
}
