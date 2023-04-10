import time
from math import inf

from src.params import VisionParams
from src.orientation_finder import OrientMethod
from src.utils import cost
from src.tester import Tester

min_cost = inf
num_iter = 500
num_refs = 8
ref_angles = {360*i/num_refs for i in range(num_refs)}

start_time = time.time()
for i in range(num_iter):
    params = VisionParams.construct_random()
    time_mean, time_std, error_mean, error_std = Tester(
        params, OrientMethod.RECOVER_POSE, ref_angles
    ).performance()
    iter_cost = cost(time_mean, time_std, error_mean, error_std)
    if iter_cost < min_cost:
        min_cost = iter_cost
        with open("best_random_params.txt", "w") as params_file:
            params_file.write(f"{params}\n")
            params_file.write(f"Time:  {time_mean:.0f}±{time_std:.0f} ms\n")
            params_file.write(f"Error: {error_mean:.2f}±{error_std:.2f} º\n")
end_time = time.time()
print(f"{num_iter} iterations in {end_time - start_time} seconds.")
