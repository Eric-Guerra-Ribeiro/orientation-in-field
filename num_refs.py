from src.params import VisionParams
from src.tester import Tester
from src.orientation_finder import OrientMethod

def method_str(method:OrientMethod):
    if method == OrientMethod.RECOVER_POSE:
        return "returnpose"
    if method == OrientMethod.BEST_REF:
        return "bestref"

num_refs = [1, 2, 3, 4, 6, 8, 12, 24]

times_mean = []
times_std = []

errors_mean = []
errors_std = []

# method = OrientMethod.BEST_REF
method = OrientMethod.RECOVER_POSE

params = VisionParams.default()

for num_ref in num_refs:
    ref_angles = {360*i/num_ref for i in range(num_ref)}
    time_mean, time_std, error_mean, error_std = Tester(
        params, method, ref_angles, True, "test"
    ).performance()
    times_mean.append(time_mean)
    times_std.append(time_std)
    errors_mean.append(error_mean)
    errors_std.append(error_std)

with open(f"num_refs_{method_str(method)}.txt", "w") as f:
    f.writelines(map(lambda num: f"{num}\n", num_refs))
with open(f"times_mean_{method_str(method)}.txt", "w") as f:
    f.writelines(map(lambda num: f"{num}\n", times_mean))
with open(f"times_std_{method_str(method)}.txt", "w") as f:
    f.writelines(map(lambda num: f"{num}\n", times_std))
with open(f"errors_mean_{method_str(method)}.txt", "w") as f:
    f.writelines(map(lambda num: f"{num}\n", errors_mean))
with open(f"erros_std_{method_str(method)}.txt", "w") as f:
    f.writelines(map(lambda num: f"{num}\n", errors_std))

for i, num_ref in enumerate(num_refs):
    print(f"{num_ref} refs: {times_mean[i]}±{times_std[i]} ms; {errors_mean[i]}±{errors_std[i]}º")
