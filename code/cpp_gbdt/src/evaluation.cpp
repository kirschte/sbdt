#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "parameters.h"
#include "evaluation.h"
#include "utils.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "spdlog/spdlog.h"

typedef std::chrono::steady_clock::time_point Timer;

/* 
    Evaluation
    - also uses threads, so we should compile with "make fast"
    - runs your dataset for different pb's and writes output to results/xy.csv
*/

int Evaluation::main(int argc, char *argv[])
{
    // Set up logging for debugging
    setup_logging(static_cast<unsigned int>( spdlog::level::err ));

    std::vector<ModelParams> parameters;

    // --------------------------------------
    // define ModelParams here
    ModelParams current_params;
    current_params.nb_trees = 50;
    current_params.leaf_clipping = false;
    current_params.balance_partition = true;
    current_params.gradient_filtering = true;
    current_params.min_samples_split = 0;
    current_params.learning_rate = 0.1;
    current_params.max_depth = 6; 
    current_params.leaf_noise = GAUSS;

    parameters.push_back(current_params);
    // --------------------------------------
    // select 1 dataset here
    // DataSet *dataset = Parser::get_adult(parameters, 50000, false);
    DataSet *dataset = Parser::get_abalone(parameters, 5000, false);

    dataset->shuffle_dataset();

    // --------------------------------------
    // select privacy budgets
    // Note: pb=0 takes much much longer than dp-trees, because we're always using all samples
    std::vector<double> budgets = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5};
    // --------------------------------------

    // output file
    std::string time_string = get_time_string();
    std::string dataset_name = dataset->name;
    int dataset_length = dataset->length;
    std::string outfile_name = fmt::format("results/TODO_create_me/{}_{}.csv", dataset_name, time_string);
    std::ofstream output;
    output.open(outfile_name);
    std::cout << "evaluation, writing results to " << outfile_name << std::endl;
    output << "dataset,nb_samples,nb_trees,use_dp,real_eps,privacy_budget,mean,std" << std::endl;

    // currently we use the same folds for all budgets. Not sure whether that's good or bad.

    ModelParams param = parameters[0];

    // run the evaluations
    size_t N_REPEATS = 6;
    size_t N_SPLITS = 5;
    for(auto budget : budgets) {
        param.privacy_budget = budget;
        param.use_dp = budget != 0.;

        std::cout << dataset_name << " pb=" << budget << std::endl;

        std::vector<TrainTestSplit *> cv_inputs;
        for (size_t repeats=0; repeats<N_REPEATS; repeats++) {
            std::vector<TrainTestSplit *> cv_input = create_cross_validation_inputs(dataset, N_SPLITS);
            cv_inputs.insert(cv_inputs.end(), cv_input.begin(), cv_input.end());
        }

        Timer time_begin = std::chrono::steady_clock::now();
        
        // prepare the ressources for each thread
        std::vector<DPEnsemble> ensembles;
        for (auto split : cv_inputs) {
            if(param.scale_y){
                split->train.scale_y(param, -1, 1);
            }
            ensembles.push_back(DPEnsemble(&param) );
        }

        // for thread-based evaluation, you can use this code instead
        /*
            std::vector<std::thread> threads(cv_inputs.size());
            // threads start training on ther respective folds
            for(size_t thread_id=0; thread_id<threads.size(); thread_id++){
                threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id],
                    &(cv_inputs[thread_id]->train));
            }
            for (auto &thread : threads) {
                thread.join(); // join once done
            }
        */

        for(size_t thread_id=0; thread_id<cv_inputs.size(); thread_id++){
            ensembles[thread_id].train(&(cv_inputs[thread_id]->train));
        }

        double real_eps = ensembles[0].real_eps;

        /* compute scores */

        std::vector<double> scores;
        for (size_t ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {
            DPEnsemble *ensemble = &ensembles[ensemble_id];
            TrainTestSplit *split = cv_inputs[ensemble_id];
            
            // predict with the test set
            std::vector<double> y_pred = ensemble->predict(split->test.X);

            if(param.scale_y){
                inverse_scale_y(param, split->train.scaler, y_pred);
            }

            // compute score            
            double score = param.task->compute_score(split->test.y, y_pred, RMSE);
            std::cout << std::setprecision(9) << score << " " << std::flush;
            scores.push_back(score);
            delete split;
        } 

        // print elapsed time
        Timer time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << "  (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;

        // write mean score to file
        double mean = compute_mean(scores);
        double stdev = compute_stdev(scores, mean);
        output << fmt::format("{},{},{},{},{},{},{},{}", dataset_name, dataset_length, param.nb_trees, param.use_dp,
            real_eps, param.privacy_budget, mean, stdev) << std::endl;

        std::cout << "mean: " << mean << std::endl;
        std::cout << "stddev: " << stdev << std::endl;
    }

    delete dataset;

    output.close();
    return 0;
}