#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include <complex>
#include <vector>
#include <fstream>
#include <iomanip>
#include "dp_tree.h"
#include "parameters.h"
#include "data.h"
#include "evaluation.h"
#include "logging.h"
#include "spdlog/spdlog.h"

class DPEnsemble
{
public:
    // constructors
    DPEnsemble(ModelParams *params);
    ~DPEnsemble();

    // fields
    std::vector<DPTree> trees;
    double init_score;

    // methods
    void train(DataSet *dataset);
    std::vector<double> predict(VVD &X) const;
    std::vector<double> predict(VVD &X, size_t treestop_early) const;
    std::vector<double> predict_without_lr_init(VVD &X) const;
    //std::vector<double> predict(VVD &X, int max_tree_idx) const;
    std::vector<double> predict_cached(VVD &X) const;
    void getTrees(VVI &split_attr, VVD &split_val,
                  std::vector<VVI> &next_node, VVD &leafs);
    void exportTrees(VVD &value, VVI &children_left, VVI &children_right,
                    VVD &threshold, std::vector<std::vector<std::complex<double>>> &impurity, VVI &feature, VVI &n_node_samples);

private:
    // fields
    ModelParams *params;
    TreeParams *tree_params = nullptr;
    DataSet *dataset;

    // methods
    void update_gradients(DataSet *dataset, std::vector<double> &gradients, int tree_index, bool cache_prediction);
    void update_hessians(DataSet *dataset, std::vector<double> &hessians);
    void update_splits(int tree_index);
public:
    double real_eps;
    double alpha;
    double max_rho;
    double leaf_sigma;
};

#endif // DPTREEENSEMBLE_H