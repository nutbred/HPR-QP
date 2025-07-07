import HPRQP_QAP_LASSO
using Printf

params = HPRQP_QAP_LASSO.HPRQP_parameters()
params.max_iter = typemax(Int32)
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0
params.problem_type = "LASSO" # "QAP" or "LASSO"

file = "xxx"  # Replace with the path to your LASSO/QAP file
result = HPRQP_QAP_LASSO.run_file(file, params)