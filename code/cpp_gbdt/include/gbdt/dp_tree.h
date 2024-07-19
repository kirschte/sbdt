#ifndef DIFFPRIVTREE_H
#define DIFFPRIVTREE_H

#include <complex>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include "tree_node.h"
#include "parameters.h"
#include "data.h"
#include "utils.h"
#include "laplace.h"
#include "logging.h"
#include "spdlog/spdlog.h"

// wrapper around attributes that represent one possible split
struct SplitCandidate {
    int feature_index;
    double split_value;
    double gain;  // gain prepared for ExpMech
    double orig_gain;  // unaltered gain
    double bucket_width;  // width of the bucket cf. ExpQ for details
    int maximize;  // -1 or 1
    int lhs_size, rhs_size;
    SplitCandidate(
        int f, double s, double g, double orig_g, double b_width, bool maxiz, int lhs_size, int rhs_size
    ) : feature_index(f), split_value(s), gain(g), orig_gain(orig_g), bucket_width(b_width), maximize(maxiz*2-1), lhs_size(lhs_size), rhs_size(rhs_size) {};
};


class DPTree
{
private:
    // fields
    ModelParams *params;
    TreeParams *tree_params;
    DataSet *dataset;
    size_t tree_index;
    std::vector<TreeNode *> leaves;
    size_t max_features;

    // methods
    TreeNode *make_tree_DFS(int current_depth, std::vector<int> live_samples, std::vector<int> live_attr, std::pair<int, int> active_split_range);
    TreeNode *make_leaf_node(int current_depth, std::vector<int> &live_samples);
    double _predict(std::vector<double> *row, TreeNode *node) const;
    TreeNode *find_best_split(VVD &X_live, std::vector<double> &gradients_live, double sum_grads_live,
                std::vector<int> &attr_live, int current_depth, std::pair<int, int> active_split_range);
    int samples_left_right_partition(std::vector<char> &lhs, VVD &samples,
                int feature_index, double feature_value, bool categorical);
    double compute_gain(std::vector<double> &samples, std::vector<double> &gradients_live, double sum_grads_live,
                double &sum_grads_lhs, double feature_value, int &lhs_size, int &categorical_skip, bool categorical);
    double compute_gain_legacy(VVD &samples, std::vector<double> &gradients_live, int feature_index,
                double feature_value, int &lhs_size, bool categorical);
    int exponential_mechanism(std::vector<SplitCandidate> &probs) {
        return exponential_mechanism(probs, false);
    };
    int exponential_mechanism(std::vector<SplitCandidate> &probs, const bool disable_dp);
    void add_leaf_noise(double eps_leaf);

public:
    // constructors
    DPTree(ModelParams *params, TreeParams *tree_params, DataSet *dataset, size_t tree_index);
    ~DPTree();

    // fields
    TreeNode *root_node;

    // methods
    void predict(VVD &X, std::vector<double> &predictions) const;
    void fit();
    void recursive_print_tree(TreeNode* node);
    void delete_tree(TreeNode *node);
    static void getTree(TreeNode* node, std::vector<int> &split_attr, std::vector<double> &split_val,
                        std::vector<std::vector<int>> &next_node, std::vector<double> &leafs);
    static void exportTree(TreeNode* node, std::vector<double> &value, std::vector<int> &children_left,
                std::vector<int> &children_right, std::vector<double> &threshold, int node_id, double l2_lambda,
                std::vector<std::complex<double>> &impurity, std::vector<int> &feature, std::vector<int> &n_node_samples);
};

#endif // DIFFPRIVTREE_H
