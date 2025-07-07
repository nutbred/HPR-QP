import HPRQP_QAP_LASSO
using Printf


params = HPRQP_QAP_LASSO.HPRQP_parameters()
params.max_iter = typemax(Int32)
params.time_limit = 3600
params.stoptol = 1e-8
params.warm_up = true
params.device_number = 0
params.problem_type = "QAP" # "QAP" or "LASSO"

data_path = "xxx"  # Path to the directory containing the dataset files
result_path = "xxx" # Path to save the results

HPRQP_QAP_LASSO.run_dataset(data_path, result_path, params)
