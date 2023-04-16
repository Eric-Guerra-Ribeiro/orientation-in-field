from src.params import VisionParams
from src.tester import Tester
from src.orientation_finder import OrientMethod

method = OrientMethod.RECOVER_POSE

params = VisionParams(1230, 1.8847186328577767, 44, 18, 0.9827160798859101, 7)

ref_angles = {45*i for i in range(8)}

tester = Tester(params, method, ref_angles, False, "test")

time_mean, time_std, error_mean, error_std = tester.performance(True, "irl_recoverpose")

print(f"recover pose: {time_mean}±{time_std} ms; {error_mean}±{error_std}º")
print(f"Fails: {tester.fails}")

method = OrientMethod.BEST_REF

time_mean, time_std, error_mean, error_std = Tester(
    params, method, ref_angles, False, "test"
).performance(True, "irl_bestref")

print(f"best ref: {time_mean}±{time_std} ms; {error_mean}±{error_std}º")
print(f"Fails: {tester.fails}")
