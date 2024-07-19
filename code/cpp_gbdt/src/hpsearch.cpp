# include <numeric>
# include <vector>
# include <iostream>
# include <iomanip>
# include <chrono>
# include <algorithm>
# include <thread>
# include <cstring>
# include "parameters.h"
# include "hpsearch.h"
# include "utils.h"
# include "gbdt/dp_ensemble.h"
# include "dataset_parser.h"
# include "data.h"
# include "loss.h"
# include "spdlog/spdlog.h"

typedef std::chrono::steady_clock::time_point Timer;

using namespace std;

vector<HyperParams> generate_combinations_A(const vector<double> g_star, const vector<double> h_star, const vector<int> nb, const vector<int> d,
const vector<double> Q_A, const vector<double> r1_A, const vector<bool> isc, const vector<bool> cfi, const vector<bool> rsc,
const vector<int> lrm, const vector<double> lam, const vector<int> rs_r, const vector<double> lr, const vector<bool> is_A,
const vector<double> isr, const vector<double> ist) {
    vector<HyperParams> combinations;
    for (double g : g_star) {
        for (double h : h_star) {
            for (int n : nb) {
                for (int dim : d) {
                    for (double q : Q_A) {
                        for (double r : r1_A) {
                            for (bool is : isc) {
                                for (bool cf : cfi) {
                                    for (bool rs : rsc) {
                                      for (int lrm_ : lrm) {
                                        for (double la : lam) {
                                          for (double l : lr) {
                                            for (bool is_ : is_A) {
                                              for (double isr_ : isr) {
                                                for (double ist_ : ist) {
                                                  combinations.push_back({g, h, n, dim, q, r, is, cf, rs, lrm_, la, l, false, -1, is_, isr_, ist_});
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return combinations;
}

vector<HyperParams> generate_combinations_B(const vector<double> g_star, const vector<double> h_star, const vector<int> nb, const vector<int> d,
const vector<double> Q_B, const vector<double> r1_B, const vector<bool> isc, const vector<bool> cfi, const vector<bool> rsc,
const vector<int> lrm, const vector<double> lam, const vector<bool> rs_B, const vector<int> rs_r, const vector<double> lr) {
    vector<HyperParams> combinations;
    for (double g : g_star) {
        for (double h : h_star) {
            for (int n : nb) {
                for (int dim : d) {
                    for (double q : Q_B) {
                        for (double r : r1_B) {
                            for (bool is : isc) {
                                for (bool cf : cfi) {
                                    for (bool rs : rsc) {
                                      for (int lrm_ : lrm) {
                                        for (double la : lam) {
                                          for (bool rs_val : rs_B) {
                                              for (int rs_r_val : rs_r) {
                                                for (double l : lr) {
                                                  combinations.push_back({g, h, n, dim, q, r, is, cf, rs, lrm_, la, l, rs_val, rs_r_val, false, 0.0, 0.0});
                                                }
                                              }
                                          }
                                        }
                                      }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return combinations;
}

int HpSearch::main(int argc, char *argv[])
{

  std::string dataset_name;
  vector<double> eps;
  max_feature_type split_method;
  bool newton_boosting = false;
  pair<double, double> feature_val_border;
  vector<double> g_star;
  vector<double> h_star;
  vector<int> nb;
  vector<int> d;
  vector<double> Q_A;
  vector<double> Q_B;
  vector<double> r1_A;
  vector<double> r1_B;
  vector<bool> isc;
  vector<bool> cfi;
  vector<bool> rsc;
  vector<int> lrm;
  vector<double> lam;
  vector<double> lr;
  vector<bool> rs_B;
  vector<int> rs_r;
  vector<bool> is_A;
  vector<double> isr;
  vector<double> ist;

  if (argc >= 3) {
    if (!std::strcmp(argv[2], "--figure3a")){
      dataset_name = "abalone";
      split_method = RAND;
      eps = {0.05, 0.1, 0.2, 0.3, 0.5};
      newton_boosting = false;
      feature_val_border = {0.0, 0.5};
      g_star = {0.1, 0.3, 0.5, 0.7};
      h_star = {0.0};
      nb = {5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600};
      d = {2, 3, 5, 6};
      Q_A = {0.1, 1.0}; 
      Q_B = {1.0};
      r1_A = {0.2, 0.5}; 
      r1_B = {0.5};
      isc = {true, false};
      cfi = {true, false};
      rsc = {true, false};
      lrm = {0, 1};
      lam = {1, 15};
      lr = {0.1, 0.2, 0.3};
      rs_B = {true, false};
      rs_r = {5, 30, 100};
      is_A = {true, false};
      isr = {0.1, 0.3};
      ist = {0.1, 0.5, 1.0};
    } else if (!std::strcmp(argv[2], "--figure3b")){
      dataset_name = "adult";
      split_method = RAND;
      eps = {0.01, 0.02, 0.03, 0.1, 0.5};
      newton_boosting = true;
      feature_val_border = {0.0, 100.0};
      g_star = {0.1, 0.3, 0.5, 0.7};
      h_star = {0.1, 0.25};
      nb = {5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600};
      d = {2, 3, 5, 6};
      Q_A = {0.005, 1.0};
      Q_B = {1.0};
      r1_A = {0.1, 0.5};
      r1_B = {0.5};
      isc = {true, false};
      cfi = {true, false};
      rsc = {true, false};
      lrm = {0, 1};
      lam = {1, 10};
      lr = {0.1, 0.2, 0.3};
      rs_B = {true, false};
      rs_r = {5, 30, 100};
      is_A = {true, false};
      isr = {0.0};
      ist = {0.0};
    } else if (!std::strcmp(argv[2], "--figure3c")){
      dataset_name = "spambase";
      split_method = RAND;
      eps = {0.01, 0.02, 0.03, 0.1, 0.5};
      newton_boosting = true;
      feature_val_border = {0.0, 1.0};
      g_star = {0.1, 0.3, 0.5, 0.7};
      h_star = {0.1, 0.25};
      nb = {5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600};
      d = {2, 3, 5, 6};
      Q_A = {0.1, 1.0};
      Q_B = {1.0};
      r1_A = {0.1, 0.2, 0.5};
      r1_B = {0.5};
      isc = {true, false};
      cfi = {true, false};
      rsc = {true, false};
      lrm = {0, 1};
      lam = {1, 15};
      lr = {0.1, 0.2, 0.3};
      rs_B = {true, false};
      rs_r = {5, 30, 100};
      is_A = {true, false};
      isr = {0.0};
      ist = {0.0};
    } else if (!std::strcmp(argv[2], "--figure3a-non-private") or !std::strcmp(argv[2], "--figure3a-non-private-random")){
      dataset_name = "abalone";
      split_method = !std::strcmp(argv[2], "--figure3a-non-private-random") ? RAND : ALL;
      eps = {0.0};
      newton_boosting = false;
      feature_val_border = {0.0, 0.0};
      g_star = {0.0};
      h_star = {0.0};
      nb = {5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600};
      d = {2, 3, 5, 6};
      Q_A = {0.1, 1.0}; 
      Q_B = {};
      r1_A = {-1.0}; 
      r1_B = {};
      isc = {false};
      cfi = {false};
      rsc = {false};
      lrm = {0, 1};
      lam = {1, 15};
      lr = {0.1, 0.2, 0.3};
      rs_B = {};
      rs_r = {};
      is_A = {false};
      isr = {0.0};
      ist = {0.0};
    } else if (!std::strcmp(argv[2], "--figure3b-non-private") or !std::strcmp(argv[2], "--figure3b-non-private-random")){
      dataset_name = "adult";
      split_method = !std::strcmp(argv[2], "--figure3b-non-private-random") ? RAND : ALL;
      eps = {0.0};
      newton_boosting = true;
      feature_val_border = {0.0, 0.0};
      g_star = {0.0};
      h_star = {0.0};
      nb = {5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600};
      d = {2, 3, 5, 6};
      Q_A = {0.005, 1.0}; 
      Q_B = {};
      r1_A = {-1.0}; 
      r1_B = {};
      isc = {false};
      cfi = {false};
      rsc = {false};
      lrm = {0, 1};
      lam = {1, 10};
      lr = {0.1, 0.2, 0.3};
      rs_B = {};
      rs_r = {};
      is_A = {false};
      isr = {0.0};
      ist = {0.0};
    } else {
      throw std::runtime_error("Unknown figure.");
    }
  }

  // Set up logging for debugging
  spdlog::set_level(spdlog::level::err);
  spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

  std::vector<ModelParams> parameters;
  ModelParams current_params;
  parameters.push_back(current_params);

  // setup of parallelization approach
  // parallel or sequential
  std::string parallelization = "parallel";

  // setup of method
  // sgbdt, maddock, xgboost or dpboost
  std::string method = "xgboost";

  // setup of learning
  // regular or stream
  bool learning_on_streams = false;
  

  if(argc > 2){
    for (int i=1; i<argc; i++) {
      std::cout << argv[i] << std::endl;
    }
  }
  
  DataSet *dataset;
  if (dataset_name == "abalone") {
    dataset = Parser::get_abalone(parameters, 5000, false);
  } else if (dataset_name == "adult") {
    dataset = Parser::get_adult(parameters, 50000, false);
  } else if (dataset_name == "spambase") {
    dataset = Parser::get_spambase(parameters, 5000, false);
  }
  dataset->shuffle_dataset();

  // output file

  std::string time_string = get_time_string();
  std::string dataset_identifier = dataset->name;
  int dataset_length = dataset->length;
  std::string outfile_name = fmt::format("results/{}/{}_{}.csv", dataset_name, dataset_identifier, time_string);
  std::ofstream output;
  output.open(outfile_name);
  std::cout << "evaluation, writing results to " << outfile_name << std::endl;
  output << "method,dataset,runs,nb_samples,learning_streams,learning_rate,nb_trees,max_depth,use_dp,real_eps,alpha,rho,sigma,privacy_budget,train_auc_mean,train_auc_std,train_accuracy_mean,train_accuracy_std,train_untuned_accuracy_mean,train_untuned_accuracy_std,train_rmse_mean,train_rmse_std,test_auc_mean,test_auc_std,test_accuracy_mean,test_accuracy_std,test_untuned_accuracy_mean,test_untuned_accuracy_std,test_rmse_mean,test_rmse_std,r1,glc,gdf,bal_p,l2_thr,hess_l2_thr,Q,use_pf,pf_add,add_trees,pf_l2_thr,pf_hess_l2_thr,pf_Q_factor,lambda,feature_val_u_border,init_ratio,init_thr,newton,cyclical_fi,refine_splits,rs_rounds,rs_subsample,ignore_split_constraints,random_splits_from_candidates,lambda_reg_mode,cut_off_leaf_denom,reg_delta,min_samples_split,split_pb_ratio" << std::endl;

  std::vector<HyperParams> combinations_A = generate_combinations_A(g_star, h_star, nb, d, Q_A, r1_A, isc, cfi, rsc, lrm, lam, rs_r, lr, is_A, isr, ist);
  std::vector<HyperParams> combinations_B = generate_combinations_B(g_star, h_star, nb, d, Q_B, r1_B, isc, cfi, rsc, lrm, lam, rs_B, rs_r, lr);

  std::vector<int> experiment_order_A(combinations_A.size());
  std::vector<int> experiment_order_B(combinations_B.size());
  std::iota(experiment_order_A.begin(), experiment_order_A.end(), 0);
  std::iota(experiment_order_B.begin(), experiment_order_B.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(experiment_order_A.begin(), experiment_order_A.end(), g);
  std::shuffle(experiment_order_B.begin(), experiment_order_B.end(), g);  

  int experiment_cnt = 0;
  size_t N_REPEATS = 4;
  size_t N_SPLITS = 5;

  while(experiment_cnt < max(combinations_A.size(), combinations_B.size())) {
    for (auto e : eps) {
      auto hp_A = combinations_A[experiment_order_A[experiment_cnt < experiment_order_A.size() ? experiment_cnt : 0]];
      auto hp_B = combinations_B.size() > 0 ?  combinations_B[experiment_order_B[experiment_cnt < experiment_order_B.size() ? experiment_cnt : 0]] : HyperParams();
      ++experiment_cnt;

      for (auto hp : std::vector<HyperParams>{hp_A, hp_B}) {

        if ((hp == hp_A && experiment_cnt >= combinations_A.size()) || (hp == hp_B && experiment_cnt >= combinations_B.size())) continue;

        ModelParams param = parameters[0];

        param.cyclical_feature_interactions = hp.cfi;
        param.newton_boosting = newton_boosting;
        param.ignore_split_constraints = hp.isc;
        param.refine_splits = hp.rs;
        param.refine_splits_rounds = hp.rs_r;
        param.refine_splits_subsample = 1.0;
        param.random_splits_from_candidates = hp.rsc;
        param.lambda_reg_mode = hp.lrm == 0 ? MAX : ADD;
        param.cut_off_leaf_denom = true;

        param.privacy_budget = e;
        param.nb_trees = hp.nb;
        param.max_depth = hp.d;
        param.scale_y = true;
        param.leaf_noise = GAUSS; 
        param.use_dp = param.privacy_budget != 0.;
        param.subsampling_ratio = hp.Q;
        param.l2_lambda = hp.lam; 
        param.feature_val_border = feature_val_border;
        param.l2_threshold = hp.g; 
        param.hess_l2_threshold = hp.h;
        param.balance_partition = false;
        param.gradient_filtering = false;
        param.leaf_clipping = false;
        param.min_samples_split = 0;
        param.learning_rate = hp.lr;
        param.max_features = split_method;
        param.max_feature_values = split_method;
        param.criterion = XGBOOST;
        param.privacy_budget_init_score_ratio = hp.is ? hp.isr : 0.0;
        param.init_score_threshold = hp.is ? hp.ist : 0.0;
        param.privacy_budget_gain_ratio = 0.0;
        param.leaf_denom_noise_weight = hp.r1;
        param.reg_delta = 2.0;
        param.additional_nb_trees = 0;

        param.continuous_learning = learning_on_streams;

        param.use_privacy_filter = false;
        param.approximate_privacy_filter = true;
        param.pf_additional_nb_trees = 0;
        param.pf_l2_threshold = param.l2_threshold;
        param.pf_hess_l2_threshold = param.hess_l2_threshold;
        param.pf_subsampling_ratio_factor = 1.0;

        if (param.use_privacy_filter && param.pf_additional_nb_trees == 0) continue;
        if (!param.use_privacy_filter && param.pf_additional_nb_trees > 0) continue;
        if (!param.use_privacy_filter && param.pf_l2_threshold != param.l2_threshold) continue;
        if (!param.use_privacy_filter && param.pf_hess_l2_threshold != param.hess_l2_threshold) continue;
        if (!param.use_privacy_filter && param.pf_subsampling_ratio_factor != 1.0) continue;
        if (param.refine_splits && !param.random_splits_from_candidates) continue;
        if (param.refine_splits && param.refine_splits_rounds == 0) continue;
        if (param.use_privacy_filter && param.refine_splits) continue; // these features together are not yet implemented
        
        std::cout << dataset_identifier << " cnt=" << experiment_cnt << " pb=" << e << std::endl;

        std::vector<TrainTestSplit *> cv_inputs;
        for (size_t repeats=0; repeats<N_REPEATS; repeats++) {
          std::vector<TrainTestSplit *> cv_input = create_cross_validation_inputs(dataset, N_SPLITS);
          cv_inputs.insert(cv_inputs.end(), cv_input.begin(), cv_input.end());
        }

        Timer time_begin = std::chrono::steady_clock::now();
      
        // prepare the ressources for each thread
        std::vector<std::thread> threads(cv_inputs.size());
        std::vector<DPEnsemble> ensembles;
        for (auto split : cv_inputs) {
          if(param.scale_y){
            split->train.scale_y(param, -1, 1);
          }
          ensembles.push_back(DPEnsemble(&param) );
        }

        std::vector<DataSet> train;
        if (method == "dpboost") {
          for (size_t thread_id=0; thread_id<cv_inputs.size(); thread_id++) {
            train.push_back(cv_inputs[thread_id]->train.copy());
          }
        }

        if (parallelization == "parallel") {
          // threads start training on ther respective folds
          for(size_t thread_id=0; thread_id<threads.size(); thread_id++){
            if (method == "dpboost") {
              threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(train[thread_id]));
            } else {
              threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(cv_inputs[thread_id]->train));
            }
          }
          for (auto &thread : threads) {
            thread.join(); // join once done
          }
        } else if (parallelization == "sequential") {
          for(size_t thread_id=0; thread_id<cv_inputs.size(); thread_id++){
            if (method == "dpboost") {
              threads[thread_id] = std::thread(&DPEnsemble::train, &ensembles[thread_id], &(train[thread_id]));
            } else {
              ensembles[thread_id].train(&(cv_inputs[thread_id]->train));
            }
          }
        }

        const double real_eps = ensembles[0].real_eps;
        const double alpha = method != "dpboost" ? ensembles[0].alpha : 0.0;
        const double rho = method != "dpboost" ? ensembles[0].max_rho : 0.0;
        const double sigma = method != "dpboost" ? ensembles[0].leaf_sigma : 0.0;

        std::cout << "sigma=" << sigma << std::endl;
        std::cout << "real_eps=" << real_eps << std::endl;

        std::vector<double> train_auc_list = {};
        std::vector<double> train_accuracy_list = {};
        std::vector<double> train_untuned_accuracy_list = {};
        std::vector<double> train_rmse_list = {};
        std::vector<double> test_auc_list = {};
        std::vector<double> test_accuracy_list = {};
        std::vector<double> test_untuned_accuracy_list = {};
        std::vector<double> test_rmse_list = {};
        for (size_t ensemble_id = 0; ensemble_id < ensembles.size(); ensemble_id++) {
          DPEnsemble *ensemble = &ensembles[ensemble_id];
          TrainTestSplit *split = cv_inputs[ensemble_id];

          // predict with the train/test set
          std::vector<double> train_y_pred = ensemble->predict(split->train.X);
          std::vector<double> test_y_pred = ensemble->predict(split->test.X);

          if(param.scale_y){
            inverse_scale_y(param, split->train.scaler, split->train.y);
            inverse_scale_y(param, split->train.scaler, train_y_pred);
            inverse_scale_y(param, split->train.scaler, test_y_pred);
          }

          // compute score
          if (dataset_name != "abalone" && dataset_name != "cal_housing" && dataset_name != "ecg_age") {

            // sigmoid
            std::transform(train_y_pred.begin(), train_y_pred.end(), train_y_pred.begin(), [](double &c){return 1./(1.+std::exp(-c));});
            std::transform(test_y_pred.begin(), test_y_pred.end(), test_y_pred.begin(), [](double &c){return 1./(1.+std::exp(-c));});

            const double train_auc = param.task->compute_score(split->train.y, train_y_pred, AUC);
            const double test_auc = param.task->compute_score(split->test.y, test_y_pred, AUC);
            const double train_accuracy = param.task->compute_score(split->train.y, train_y_pred, ACC);
            const double train_untuned_accuracy = param.task->compute_score(split->train.y, train_y_pred, UNTUNED_ACC);
            const double test_accuracy = param.task->compute_score(split->test.y, test_y_pred, ACC);
            const double test_untuned_accuracy = param.task->compute_score(split->test.y, test_y_pred, UNTUNED_ACC);
            train_auc_list.push_back(train_auc);
            test_auc_list.push_back(test_auc);
            train_accuracy_list.push_back(train_accuracy);
            train_untuned_accuracy_list.push_back(train_untuned_accuracy);
            test_accuracy_list.push_back(test_accuracy);
            test_untuned_accuracy_list.push_back(test_untuned_accuracy);
            std::cout << std::setprecision(9) << test_auc << " " << test_accuracy << " " << std::flush;
          } else {
            const double train_rmse = param.task->compute_score(split->train.y, train_y_pred, RMSE);
            const double test_rmse = param.task->compute_score(split->test.y, test_y_pred, RMSE);
            train_rmse_list.push_back(train_rmse);
            test_rmse_list.push_back(test_rmse);
            std::cout << std::setprecision(9) << test_rmse << " " << std::flush;
          }
          
          delete split;
        }

        // print elapsed time
        Timer time_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (time_end - time_begin).count();
        std::cout << " (" << std::fixed << std::setprecision(1) << elapsed/1000 << "s)" << std::endl;

        // write mean score to file
        double train_auc_mean = 0.0;
        double train_auc_stdev = 0.0;
        double train_accuracy_mean = 0.0;
        double train_accuracy_stdev = 0.0;
        double train_untuned_accuracy_mean = 0.0;
        double train_untuned_accuracy_stdev = 0.0;
        double train_rmse_mean = 0.0;
        double train_rmse_stdev = 0.0;
        double test_auc_mean = 0.0;
        double test_auc_stdev = 0.0;
        double test_accuracy_mean = 0.0;
        double test_accuracy_stdev = 0.0;
        double test_untuned_accuracy_mean = 0.0;
        double test_untuned_accuracy_stdev = 0.0;
        double test_rmse_mean = 0.0;
        double test_rmse_stdev = 0.0;
        
        if (dataset_name != "abalone" && dataset_name != "cal_housing" && dataset_name != "ecg_age") {
          train_auc_mean = compute_mean(train_auc_list);
          train_auc_stdev = compute_stdev(train_auc_list, train_auc_mean);
          train_accuracy_mean = compute_mean(train_accuracy_list);
          train_accuracy_stdev = compute_stdev(train_accuracy_list, train_accuracy_mean);
          train_untuned_accuracy_mean = compute_mean(train_untuned_accuracy_list);
          train_untuned_accuracy_stdev = compute_stdev(train_untuned_accuracy_list, train_untuned_accuracy_mean);
          test_auc_mean = compute_mean(test_auc_list);
          test_auc_stdev = compute_stdev(test_auc_list, test_auc_mean);
          test_accuracy_mean = compute_mean(test_accuracy_list);
          test_accuracy_stdev = compute_stdev(test_accuracy_list, test_accuracy_mean);
          test_untuned_accuracy_mean = compute_mean(test_untuned_accuracy_list);
          test_untuned_accuracy_stdev = compute_stdev(test_untuned_accuracy_list, test_untuned_accuracy_mean);
        } else {
          train_rmse_mean = compute_mean(train_rmse_list);
          train_rmse_stdev = compute_stdev(train_rmse_list, train_rmse_mean);
          test_rmse_mean = compute_mean(test_rmse_list);
          test_rmse_stdev = compute_stdev(test_rmse_list, test_rmse_mean);
        }
        
        output <<
          method << "," << dataset_identifier << "," << N_SPLITS * N_REPEATS << "," << dataset_length << "," << param.continuous_learning << "," << param.learning_rate << "," << param.nb_trees << "," << param.max_depth << "," << param.use_dp << ","
          << real_eps << "," << alpha << "," << rho << "," << sigma << "," << param.privacy_budget << "," << 
          train_auc_mean << "," << train_auc_stdev << "," << 
          train_accuracy_mean << "," << train_accuracy_stdev << "," << 
          train_untuned_accuracy_mean << "," << train_untuned_accuracy_stdev << "," <<
          train_rmse_mean << "," << train_rmse_stdev << "," <<
          test_auc_mean << "," << test_auc_stdev << "," << 
          test_accuracy_mean << "," << test_accuracy_stdev << "," << 
          test_untuned_accuracy_mean << "," << test_untuned_accuracy_stdev << "," <<
          test_rmse_mean << "," << test_rmse_stdev << "," <<
          param.leaf_denom_noise_weight << "," << param.leaf_clipping << "," << param.gradient_filtering << "," << param.balance_partition << "," <<
          param.l2_threshold << "," <<param.hess_l2_threshold << "," <<param.subsampling_ratio << "," <<param.use_privacy_filter << "," <<
          param.pf_additional_nb_trees << "," << param.additional_nb_trees << "," <<param.pf_l2_threshold << "," << param.pf_hess_l2_threshold << "," <<param.pf_subsampling_ratio_factor << "," << 
          param.l2_lambda << "," << param.feature_val_border.second << "," << param.privacy_budget_init_score_ratio << "," << param.init_score_threshold << "," << 
          param.newton_boosting << "," << param.cyclical_feature_interactions << "," << 
          param.refine_splits << "," << param.refine_splits_rounds << "," << param.refine_splits_subsample << "," << 
          param.ignore_split_constraints << "," << param.random_splits_from_candidates << "," << param.lambda_reg_mode << "," << param.cut_off_leaf_denom << "," << 
          param.reg_delta << "," << param.min_samples_split << "," << param.privacy_budget_gain_ratio << std::endl;

        if (dataset_name != "abalone" && dataset_name != "cal_housing" && dataset_name != "ecg_age") {
          std::cout << "auc mean: " << std::setprecision(3) << test_auc_mean << std::endl;
          std::cout << "auc stddev: " << test_auc_stdev << std::endl;

          std::cout << "accuracy mean: " << std::setprecision(3) << test_accuracy_mean << std::endl;
          std::cout << "accuracy stddev: " << test_accuracy_stdev << std::endl;
        } else {
          std::cout << "rmse mean: " << std::setprecision(3) << test_rmse_mean << std::endl;
          std::cout << "rmse stddev: " << test_rmse_stdev << std::endl;
        }
      }
    }
  }

  std::cout << "experiment_cnt=" << experiment_cnt << std::endl;

  delete dataset;
  output.close();
  return 0;
}
