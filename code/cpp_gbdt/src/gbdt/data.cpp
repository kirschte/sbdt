#include <vector>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>
#include "data.h"

Scaler::Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required) : data_min(min_val), data_max(max_val),
        feature_min(fmin), feature_max(fmax), scaling_required(scaling_required)
{
    this->minimum_y = min_val;
    this->maximum_y = max_val;
    this->lower = fmin;
    this->upper = fmax;
    double data_range = data_max - data_min;
    data_range = data_range == 0 ? 1 : data_range;
    this->scale = (feature_max - feature_min) / data_range;
    this->min_ = feature_min - data_min * scale;
}


DataSet::DataSet()
{
    empty = true;
}

DataSet::DataSet(VVD X, std::vector<double> y, std::string name) : DataSet(X, y)
{
    this->name = name;
}


DataSet::DataSet(VVD X, std::vector<double> y) : X(X), y(y)
{
    if(X.size() != y.size()){
        std::stringstream message;
        message << "X,y need equal amount of rows! (" << X.size() << ',' << y.size() << ')';
        throw std::runtime_error(message.str());
    }
    length = X.size();
    num_x_cols = X[0].size();
    empty = false;
    identifiers = std::vector<int>(length);
    std::iota(identifiers.begin(), identifiers.end(), 0);
}


// scale y values to be in [lower,upper]
void DataSet::scale_y(ModelParams &params, double lower, double upper)
{
    // only scale in dp mode
    if(params.use_dp){
        
        // return if no scaling required (y already in [-1,1])
        bool scaling_required = false;
        for(auto elem : y) {
            if (elem < lower or elem > upper) {
                scaling_required = true; break;
            }
        }
        if (not scaling_required) {
            scaler = Scaler(0,0,0,0,false);
            return;
        }

        double doublemax = std::numeric_limits<double>::max();
        double doublemin = std::numeric_limits<double>::min();
        double minimum_y = doublemax, maximum_y = doublemin;
        for(int i=0; i<length; i++) {
            minimum_y = std::min(minimum_y, y[i]);
            maximum_y = std::max(maximum_y, y[i]);
        }
        for(int i=0; i<length; i++) {
            y[i] = (y[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
        }
        scaler = Scaler(minimum_y, maximum_y, lower, upper, true);
    }
}

void DataSet::scale_y_with_scaler(ModelParams &params, Scaler scaler) 
{
    // only scale in dp mode
    if(params.use_dp){
        const double minimum_y = scaler.minimum_y;
        const double maximum_y = scaler.maximum_y;
        const double lower = scaler.lower;
        const double upper = scaler.upper;
        for(int i=0; i<length; i++) {
            y[i] = (y[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
        }
    }
}

void inverse_scale_y(ModelParams &params, Scaler &scaler, std::vector<double> &vec)
{
    if(params.use_dp){
        // return if no scaling required
        if(not scaler.scaling_required){
            return;
        }

        for(auto &elem : vec) {
            elem -= scaler.min_;
            elem /= scaler.scale;
        }
    }
}

// algorithm (EXPQ) from this paper:
// https://arxiv.org/pdf/2001.02285.pdf
// corresponding code:
// https://github.com/wxindu/dp-conf-int/blob/master/algorithms/alg5_EXPQ.R
std::tuple<double,double> dp_confidence_interval(std::vector<double> &samples, double percentile, double budget)
{
    // e.g.  95% -> {0.025, 0.975}
    std::vector<double> quantiles = {(1.0-percentile/100.)/2., percentile/100. + (1.0-percentile/100.)/2.};
    std::vector<double> results;

    // set up inputs
    std::sort(samples.begin(),samples.end());
    double *db = samples.data();
    int n = samples.size();
    double e = budget / 2;  // half budget since we're doing it twice

    // run the dp quantile calculation twice (to get the lower & upper bound)
    for(auto quantile : quantiles) {

        double q = quantile;
        int qi = std::floor((n-1)*q + 1.5);
        std::vector<double> probs(n+1);
        std::iota(probs.begin(), probs.end(), 1.0);   // [1,2,...,n+1]
        double r = ((double)std::rand()/(double)RAND_MAX);
        int priv_qi;
        
        // exponential mechanism
        // https://github.com/wxindu/dp-conf-int/blob/master/algorithms/exp_mech.c
        int m = qi;
        for(int i = 0; i < m; i++) {
            double utility = (i + 1) - m;
            probs[i] = std::max(0.0, (db[i + 1] - db[i]) * std::exp(e * utility / 2.));
        }
        for(int i = m; i <= n; i++) {
            double utility = m - i;
            probs[i] = std::max(0.0, (db[i + 1] - db[i]) * std::exp(e * utility / 2.));
        }
        double sum = 0;
        for(int i = 0; i <= n; i++) sum += probs[i];
        r *= sum;
        for(int i = 0; i <= n; i++) {
            r -= probs[i];
            if(r < 0) {
                priv_qi = i;
                break;
            }
        }
        std::uniform_real_distribution<double> unif(db[priv_qi],db[priv_qi+1]);
        std::default_random_engine re;
        double a_random_double = unif(re);
        results.push_back(a_random_double);
    }
    return std::make_tuple(results[0],results[1]);
}


TrainTestSplit train_test_split_random(DataSet &dataset, double train_ratio, bool shuffle)
{
    if(shuffle) {
        dataset.shuffle_dataset();
    }

    int border = ceil((1-train_ratio) * dataset.y.size());

    VVD x_test(dataset.X.begin(), dataset.X.begin() + border);
    std::vector<double> y_test(dataset.y.begin(), dataset.y.begin() + border);
    VVD x_train(dataset.X.begin() + border, dataset.X.end());
    std::vector<double> y_train(dataset.y.begin() + border, dataset.y.end());

    if(train_ratio >= 1) {
        DataSet train(x_train, y_train);
        return TrainTestSplit(train, DataSet());
    } else if (train_ratio <= 0) {
        DataSet test(x_test, y_test);
        return TrainTestSplit(DataSet(), test);
    } else {
        DataSet train(x_train, y_train);
        DataSet test(x_test, y_test);
        return TrainTestSplit(train, test);
    }
}

// "reverse engineered" the python sklearn.model_selection.cross_val_score
// Returns a std::vector of the train-test-splits. Will by default not shuffle 
// the dataset rows.
std::vector<TrainTestSplit *> create_cross_validation_inputs(DataSet *dataset, int folds)
{
    bool shuffle = false;
    if(shuffle) {
        dataset->shuffle_dataset();
    }

    int fold_size = dataset->length / folds;
    std::vector<int> fold_sizes(folds, fold_size);
    int remainder = dataset->length % folds;
    int index = 0;
    while (remainder != 0) {
        fold_sizes[index++]++;
        remainder--;
    }
    // each entry in "indices" marks a start of the test set
    // ->     [ test |        train          ]
    //                      ...
    //        [   train..   | test |  ..train ]
    //                      ...
    std::deque<int> indices(folds);
    std::partial_sum(fold_sizes.begin(), fold_sizes.end(), indices.begin());
    indices.push_front(0); 
    indices.pop_back();

    std::vector<TrainTestSplit *> splits;

    for(int i=0; i<folds; i++) {

        // don't waste memory by using local copies of the vectors.
        // work directly on what will be used.
        TrainTestSplit *split = new TrainTestSplit();
        DataSet *train = &split->train;
        DataSet *test = &split->test;

        VVD::iterator x_iterator = (dataset->X).begin() + indices[i];
        std::vector<double>::iterator y_iterator = (dataset->y).begin() + indices[i];
        std::vector<int>::iterator cluster_ids_iterator;
        if (not dataset->cluster_ids.empty()) {
            cluster_ids_iterator = (dataset->cluster_ids).begin() + indices[i];
        }

        // extracting the test slice is easy
        test->X = VVD(x_iterator, x_iterator + fold_sizes[i]);
        test->y = std::vector<double>(y_iterator, y_iterator + fold_sizes[i]);
        if (not dataset->cluster_ids.empty()) {
            test->cluster_ids = std::vector<int>(cluster_ids_iterator, cluster_ids_iterator + fold_sizes[i]);
        }

        // building the train set from the remaining rows is slightly more tricky
        // (if you don't want to waste memory)
        if(i != 0){     // part before the test slice
            train->X = VVD((dataset->X).begin(), (dataset->X).begin() + indices[i]);
            train->y = std::vector<double>((dataset->y).begin(), (dataset->y).begin() + indices[i]);
            if (not dataset->cluster_ids.empty()) {
                train->cluster_ids = std::vector<int>((dataset->cluster_ids).begin(), (dataset->cluster_ids).begin() + indices[i]);
            }
        }
        if(i < folds-1){    //part after the test slice
            for(int cur_row = indices[i+1]; cur_row < dataset->length; cur_row++){
                train->X.push_back(dataset->X[cur_row]);
                train->y.push_back(dataset->y[cur_row]);
                if (not dataset->cluster_ids.empty()) {
                    train->cluster_ids.push_back(dataset->cluster_ids[cur_row]);
                }
            }
        }
        // don't forget to add the meta information
        train->name = dataset->name;
        train->length = train->X.size();
        train->num_x_cols = train->X[0].size();
        train->empty = false;
        test->name = dataset->name;
        test->length = test->X.size();
        test->num_x_cols = test->X[0].size();
        test->empty = false;

        splits.push_back(split);
    }
    return splits;
}


void DataSet::shuffle_dataset()
{
    std::vector<int> indices(length);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::random_shuffle(indices.begin(), indices.end());
    DataSet copy = *this;
    for(size_t i=0; i<indices.size(); i++){
        X[i] = copy.X[indices[i]];
        y[i] = copy.y[indices[i]];
        if (not gradients.empty()) {
            gradients[i] = copy.gradients[i];
        }
        if (not hessians.empty()) {
            hessians[i] = copy.hessians[i];
        }
        if (not identifiers.empty()) {
            identifiers[i] = copy.identifiers[i];
        }
        if (not cluster_ids.empty()) {
            cluster_ids[i] = copy.cluster_ids[i];
        }
    }
}


DataSet DataSet::get_subset(std::vector<int> &indices)
{
    DataSet dataset;
    std::unordered_set<int> idxs(indices.begin(), indices.end());
    for (int i=0; i<length; i++) {
        if (idxs.find(i) != idxs.end()) {  // O(1) due to hashmap
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            if (not gradients.empty()) {
                dataset.gradients.push_back(gradients[i]);
            }
            if (not hessians.empty()) {
                dataset.hessians.push_back(hessians[i]);
            }
            if (not identifiers.empty()) {
                dataset.identifiers.push_back(identifiers[i]);
            }
            if (not cluster_ids.empty()) {
                dataset.cluster_ids.push_back(cluster_ids[i]);
            }
        }
    }
    dataset.name = this->name;
    dataset.length = dataset.y.size();
    dataset.num_x_cols = this->num_x_cols;
    dataset.empty = dataset.y.empty();
    return dataset;
}

DataSet DataSet::copy()
{
    DataSet dataset;
    for (int i=0; i<length; i++) {
        dataset.X.push_back(X[i]);
        dataset.y.push_back(y[i]);
        if (not gradients.empty()) {
            dataset.gradients.push_back(gradients[i]);
        }
        if (not hessians.empty()) {
            dataset.hessians.push_back(hessians[i]);
        }
        if (not identifiers.empty()) {
            dataset.identifiers.push_back(identifiers[i]);
        }
        if (not cluster_ids.empty()) {
            dataset.cluster_ids.push_back(cluster_ids[i]);
        }
    }
    dataset.name = this->name;
    dataset.length = dataset.y.size();
    dataset.num_x_cols = this->num_x_cols;
    dataset.empty = dataset.y.empty();
    return dataset;
}

DataSet DataSet::remove_rows(std::vector<int> &indices)
{
    DataSet dataset;
    dataset.X.reserve( length - indices.size() );
    dataset.y.reserve( length - indices.size() );
    dataset.gradients.reserve( length - indices.size() );
    dataset.hessians.reserve( length - indices.size() );
    dataset.identifiers.reserve( length - indices.size() );
    dataset.cluster_ids.reserve( length - indices.size());

    std::unordered_set<int> idxs(indices.begin(), indices.end());
    for (int i=0; i<length; i++) {
        if (idxs.find(i) == idxs.end()) {  // O(1) due to hashmap
            dataset.X.push_back(X[i]);
            dataset.y.push_back(y[i]);
            if (not gradients.empty()) {
                dataset.gradients.push_back(gradients[i]);
            }
            if (not hessians.empty()) {
                dataset.hessians.push_back(hessians[i]);
            }
            if (not identifiers.empty()) {
                dataset.identifiers.push_back(identifiers[i]);
            }
            if (not cluster_ids.empty()) {
                dataset.cluster_ids.push_back(cluster_ids[i]);
            }
        }
    }
    dataset.name = this->name;
    dataset.length = dataset.y.size();
    dataset.num_x_cols = X[0].size();
    dataset.empty = dataset.length == 0;
    dataset.scaler = scaler;
    return dataset;
}
