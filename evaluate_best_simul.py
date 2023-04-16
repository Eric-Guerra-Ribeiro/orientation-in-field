from src.params import VisionParams
from src.tester import Tester
from src.orientation_finder import OrientMethod

method = OrientMethod.RECOVER_POSE

params = VisionParams(1260, 1.9575404949239472, 31, 20, 0.9980880128787601, 10)

ref_angles = {45*i for i in range(8)}

time_mean, time_std, error_mean, error_std = Tester(
    params, method, ref_angles, True, "test"
).performance(True, "recoverpose")

print(f"recover pose: {time_mean}±{time_std} ms; {error_mean}±{error_std}º")

method = OrientMethod.BEST_REF

time_mean, time_std, error_mean, error_std = Tester(
    params, method, ref_angles, True, "test"
).performance(True, "bestref")

print(f"recover pose: {time_mean}±{time_std} ms; {error_mean}±{error_std}º")
