#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "logging.h"
#include "parameters.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "evaluation.h"
#include "hpsearch.h"
#include "rerun.h"
#include "utils.h"
#include "spdlog/spdlog.h"

extern bool AVGLEAKAGE_MODE;
std::ofstream avgleakage;


int main(int argc, char** argv)
{
    // seed randomness once and for all
    srand((unsigned)time(NULL));

    // parse flags, currently supporting "--verify", "--bench", "--eval"
    if(argc != 1){
        for(int i = 1; i < argc; i++){
            if ( ! std::strcmp(argv[i], "--avgleak") ){
                // go into printing average leakage mode (only available in this main)
                AVGLEAKAGE_MODE = true;
            } else if ( ! std::strcmp(argv[i], "--eval") ){
                // go into evaluation mode
                return Evaluation::main(argc, argv); 
            } else if ( ! std::strcmp(argv[i], "--hpsearch") ){
                // hyperparameter search
                return HpSearch::main(argc, argv); 
            } else if ( ! std::strcmp(argv[i], "--rerun") ){
                // rerun top performers
                return Rerun::main(argc, argv); 
            } else {
                throw std::runtime_error("unkown command line flag encountered");
            } 
        }
    } else { // no flags given, continue in this file
        AVGLEAKAGE_MODE = false;
    }

    // Set up logging
    setup_logging(spdlog::level::err);

    // Define model parameters
    // reason to use a vector is because parser expects it
    std::vector<ModelParams> parameters;
    ModelParams current_params = create_default_params();

    parameters.push_back(current_params);

    // Choose your dataset here
    DataSet *dataset = Parser::get_abalone(parameters, 5000, false);
    // DataSet *dataset = Parser::get_adult(parameters, 50000, false);

    std::cout << dataset->name << std::endl;

    ModelParams params = parameters[0];

    std::vector<double> scores;

    // create cross validation inputs
    std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, 5);
    std::string dataset_name = dataset->name;
    delete dataset;

    // do cross validation
    int i = 0;
    for (auto split : cv_inputs) {

        if (AVGLEAKAGE_MODE) {
            std::string time_string = get_time_string();
            std::string avgleakagefile_name = fmt::format("cpp_gbdt/results/individual_leakage/{}_{}_{}.csv", dataset_name, time_string, i++);
            avgleakage.open(avgleakagefile_name);
            avgleakage << "id,kl" << std::endl;
        }

        if(params.scale_y){
            split->train.scale_y(params, -1, 1);
        }

        DPEnsemble ensemble = DPEnsemble(&params);
        ensemble.train(&split->train);
        LOG_INFO("Accounted epsilon = {1:.2f} and alpha = {2}", ensemble.real_eps, ensemble.alpha);
        
        // predict with the test set
        std::vector<double> y_pred = ensemble.predict(split->test.X);

        if(params.scale_y) {
            inverse_scale_y(params, split->train.scaler, y_pred);
        }

        // compute score
        double score = params.task->compute_score(split->test.y, y_pred, RMSE);

        std::cout << score << " " << std::flush;
        scores.push_back(score);
        delete split;
        
        if (AVGLEAKAGE_MODE) avgleakage.close();
    } std::cout << std::endl;

}
