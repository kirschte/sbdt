#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>
#include <random>


typedef enum {RMSE, ACC, UNTUNED_ACC, AUC_WMW, AUC, F1} classify_metric;

inline std::string cm_to_str(classify_metric t) {
    switch (t) {
        case RMSE: return "RMSE";
        case ACC: return "Accuracy";
        case UNTUNED_ACC: return "Untuned Accuracy";
        case AUC_WMW: return "AUC (Wilcoxon Mann Whitney)";
        case AUC: return "AUC";
        case F1: return "F1 Score";
    }
    return "undef";
}


// abstract class
class Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred, bool bce_loss) = 0;
    virtual std::vector<double> compute_hessians(std::vector<double> &y, std::vector<double> &gradients) = 0;
    virtual double compute_init_score(std::vector<double> &y, double threshold, bool use_dp, double privacy_budget) = 0;
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred, classify_metric metric) = 0;
};


// uses Least Squares as cost/loss function
class Regression : public Task
{

public:

    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred, bool /*bce_loss*/);

    virtual std::vector<double> compute_hessians(std::vector<double> &y, std::vector<double> &/*gradients*/);
    
    // mean
    virtual double compute_init_score(std::vector<double> &y, double threshold, bool use_dp, double privacy_budget);

    // RMSE
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred, classify_metric metric);

};

// uses Binomial Deviance as cost/loss function
class BinaryClassification : public Task
{
public:

    // expit
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred, bool bce_loss);
    
    virtual std::vector<double> compute_hessians(std::vector<double> &y, std::vector<double> &gradients);

    // logit
    virtual double compute_init_score(std::vector<double> &y, double threshold, bool use_dp, double privacy_budget);

    // different scores
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred, classify_metric metric);

private:

    double accuracy(std::vector<double> &y, std::vector<double> &y_pred, double threshold);
};


#endif /* LOSS_FUNCTION_H */