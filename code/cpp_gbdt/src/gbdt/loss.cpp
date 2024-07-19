#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <limits>
#include "loss.h"
#include "utils.h"
#include "laplace.h"
#include "logging.h"

/* ---------- Regression ---------- */

double Regression::compute_init_score(std::vector<double> &y, double threshold, bool use_dp, double privacy_budget)
{
    std::vector<double> clipped_y(y.size(), 0.0);
    std::transform(y.begin(), y.end(), clipped_y.begin(), [threshold](double a) { return clamp(a, -threshold, threshold); });
    double mean = std::accumulate(clipped_y.begin(), clipped_y.end(), 0.0) / clipped_y.size();
    if (use_dp) {
        std::random_device rd;
        Laplace lap(rd());
        if (privacy_budget > 0.0) {
            return mean + lap.return_a_random_variable(threshold / (clipped_y.size() * privacy_budget));
        } else {
            return 0.0;
        }
    } else {
        return mean;
    }
}

std::vector<double> Regression::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred, bool /*bce_loss*/)
{
    std::vector<double> gradients(y.size());
    for (size_t i=0; i<y.size(); i++) {
        gradients[i] = y_pred[i] - y[i];
    }
    return gradients;
}

std::vector<double> Regression::compute_hessians(std::vector<double> &y, std::vector<double> &/*gradients*/) 
{
    std::vector<double> hessians(y.size(), 0.0);
    return hessians;
}
    
double Regression::compute_score(std::vector<double> &y, std::vector<double> &y_pred, classify_metric metric)
{
    // RMSE
    if (metric != RMSE) LOG_INFO("Wrong metric. Chose \"{1}\" for regression task.", cm_to_str(metric));

    std::transform(y.begin(), y.end(), 
            y_pred.begin(), y_pred.begin(), std::minus<double>());
    std::transform(y_pred.begin(), y_pred.end(),
            y_pred.begin(), [](double &c){return std::pow(c,2);});
    double average = std::accumulate(y_pred.begin(),y_pred.end(), 0.0) / y_pred.size();
    double rmse = std::sqrt(average);
    return rmse;
}


/* ---------- Binary Classification ---------- */

// Uses Binomial Deviance

double BinaryClassification::compute_init_score(std::vector<double> &y, double threshold, bool use_dp, double privacy_budget)
{
    std::vector<double> clipped_y(y.size(), 0.0);
    std::transform(y.begin(), y.end(), clipped_y.begin(), [threshold](double a) { return clamp(a, 0.0, threshold); });
    double mean = std::accumulate(clipped_y.begin(), clipped_y.end(), 0.0) / clipped_y.size();
    if (use_dp) {
        std::random_device rd;
        Laplace lap(rd());
        if (privacy_budget > 0.0) {
            mean = mean + lap.return_a_random_variable(0.5 * threshold / (y.size() * privacy_budget));
        } else {
            return 0.0; // log(0.5 / (1 - 0.5)) = 0
        }
    }
    mean = clamp(mean, std::numeric_limits<double>::epsilon() * 10, 1 - (std::numeric_limits<double>::epsilon() * 10));
    return std::log(mean) - std::log(1.0 - mean); // log(mean / (1 - mean)); inverse of sigmoid
}

std::vector<double> BinaryClassification::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred, bool bce_loss)
{
    std::vector<double> gradients(y.size());
    for (size_t i=0; i<y.size(); i++) {
        if (bce_loss) {
            // gradient for binary cross entropy loss (or binary log loss or binomial deviance loss)
            gradients[i] = 1 / (1 + std::exp(-y_pred[i])) - y[i];
        } else {
            // gradient for MSE loss
            gradients[i] = y_pred[i] - y[i];
        }
    }
    return gradients;
}

std::vector<double> BinaryClassification::compute_hessians(std::vector<double> &y, std::vector<double> &gradients)
{   
    std::vector<double> hessians(y.size());
    for (size_t i=0; i<y.size(); i++) {
        hessians[i] = (y[i] - (-gradients[i])) * (1 - y[i] + (-gradients[i]));
    }
    return hessians;
}

double BinaryClassification::accuracy(std::vector<double> &y, std::vector<double> &y_pred, double threshold) {
    std::vector<bool> correct_preds(y.size());
    int i = 0;
    for(auto &elem : y_pred){
        double prediction = (elem < threshold) ? 0.0 : 1.0;
        correct_preds[i] = (y[i] == prediction);
        i += 1;
    }

    const double true_preds = std::count(correct_preds.begin(), correct_preds.end(), true);
    const double acc = true_preds / y.size() * 100;
    return acc;
}


double BinaryClassification::compute_score(std::vector<double> &y, std::vector<double> &y_pred, classify_metric metric)
{
    if (metric == RMSE) {
        LOG_INFO("Wrong metric. Chose \"{1}\" for classification task. Fallback to accuracy (ACC).", cm_to_str(metric));
        metric = ACC;
    }

    double max_acc = -1;
    if (metric == ACC) { // accuracy
        for (double thr=*std::min_element(y_pred.begin(), y_pred.end()); thr<=*std::max_element(y_pred.begin(), y_pred.end()); thr+=0.01) {
            const double acc = accuracy(y, y_pred, thr);
            if (acc > max_acc) max_acc = acc;
        }
        return max_acc;
    } else if (metric == UNTUNED_ACC) {
        const double untuned_accuracy = accuracy(y, y_pred, 0.5);
        return untuned_accuracy;
    }
    else if (metric == AUC_WMW) { // (for exact AUC use metric=AUC) estimate of area under curve via Wilcoxon-Mann-Whitney statistic 
        int indicator_sum = 0;
        int neg_sum = 0;
        int pos_sum = 0;
        for (size_t c1 = 0; c1 < y.size(); c1++) {
            if (y[c1] == 0) {
                neg_sum++;
            }
            if (y[c1] == 1) {
                pos_sum++;
            }
            for (size_t c2 = 0; c2 < y.size(); c2++) {
                if (y[c1] == 0 and y[c2] == 1) {
                    if (y_pred[c1] < y_pred[c2]) {
                        indicator_sum++;
                    }
                }
            }
        }
        return 100. * indicator_sum / (1. * neg_sum * pos_sum);
    } else if (metric == AUC) { // area under curve (exact)
        std::function<bool(const std::vector<double>&, const std::vector<double>&)> sortcol;
        sortcol = [](const std::vector<double>& v1, const std::vector<double>& v2) {
            return v1[0] < v2[0];
        };
        std::vector<std::vector<double>> scores;
        double step_size = 0.005;
        
        const double thr_start = *std::min_element(y_pred.begin(), y_pred.end());
        const double thr_end = *std::max_element(y_pred.begin(), y_pred.end());
        for (double thr = thr_start; thr<=thr_end; thr += step_size) {
            int TP = 0, FP = 0;
            int P = 0, N = 0;
            for (size_t c=0; c<y.size(); c++) {
                if (y[c] == 1) {
                    P++;
                } else {
                    N++;
                }
                if (y_pred[c] >= thr) {
                    if (y[c] == 1) {
                        TP++;
                    } else {
                        FP++;
                    }
                }
            }
            double TPR = (TP+0.) / P;
            double FPR = (FP+0.) / N;
            scores.push_back(std::vector<double>{FPR, TPR});
        }
        sort(scores.begin(), scores.end(), sortcol);
        scores.insert(scores.begin(), std::vector<double>{0.,0.});
        scores.push_back(std::vector<double>{1.,1.});

        double AUC = 0.0;
        for (size_t c = 0; c < scores.size() - 1; c++) {
            AUC += (scores[c+1][0] - scores[c][0]) * 0.5 * (scores[c+1][1] + scores[c][1]);
        }

        return AUC;
    } else if (metric == F1) { // f1 score
        int TP = 0, FP = 0, FN = 0;
        for (size_t c=0; c<y.size(); c++) {
            if (y_pred[c] >= 0.0) {
                if (y[c] == 1) {
                    TP++;
                } else {
                    FP++;
                }
            } else {
                if (y[c] == 1) {
                    FN++;
                }
            }
        }
        return 2.0*TP / (2.0*TP + FP + FN);
    }
    return 0.0;
}
