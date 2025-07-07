import HPRQP

params = HPRQP.HPRQP_parameters()
params.max_iter = typemax(Int32)
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0 

file = "xxx" # Replace with the path to your QP file
result = HPRQP.run_file(file, params)

println("Objective value: ", result.primal_obj)