from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

times_recoverpose = np.loadtxt("./results/best_params/times_recoverpose.txt")
errors_recoverpose = np.loadtxt("./results/best_params/errors_recoverpose.txt")

print(f"{reduce(lambda cum, val: cum + (1 if val > 45 else 0), errors_recoverpose, 0)}")
print(len(errors_recoverpose))
times_bestref = np.loadtxt("./results/best_params/times_bestref.txt")
errors_bestref = np.loadtxt("./results/best_params/errors_bestref.txt")

print(f"{reduce(lambda cum, val: cum + (1 if val > 45 else 0), errors_bestref, 0)}")
print(len(errors_bestref))

errors_irl = np.loadtxt("./results/irl/errors_irl_recoverpose.txt")

print(f"{reduce(lambda cum, val: cum + (1 if val < 5 else 0), errors_irl, 0)}")
print(len(errors_irl) + 6)

plt.rc('text' , usetex = True)
plt.rcParams.update({ 'font.family': 'serif' })
plt.rcParams.update({'font.size': 18})

plt.figure()
# plt.grid()
plt.hist(errors_recoverpose, bins=np.linspace(0, max(errors_recoverpose), int(max(errors_recoverpose)/5)),
         histtype='bar', ec='black', color="tab:orange")
plt.title('Error Histogram - Recover Pose')
plt.xlabel('Angle Error ($^o$)')
plt.ylabel('Frequency')
plt.locator_params(axis='y', integer=True)
plt.tight_layout ()
plt.savefig ('best_params_error_recoverpose.eps', format = 'eps')
plt.savefig ('best_params_error_recoverpose.png', format = 'png' , dpi=400)

plt.figure()
# plt.grid()
plt.hist(times_recoverpose, bins=np.linspace(0, max(times_recoverpose), int(max(times_recoverpose)/5)),
         histtype='bar', ec='black', color="tab:orange")
plt.title('Processing Time Histogram - Recover Pose')
plt.xlabel('Processing Time (ms)')
plt.ylabel('Frequency')
plt.locator_params(axis='y', integer=True)
plt.tight_layout ()
plt.savefig ('best_params_time_recoverpose.eps', format = 'eps')
plt.savefig ('best_params_time_recoverpose.png', format = 'png' , dpi=400)

plt.figure()
# plt.grid()
plt.hist(errors_bestref, bins=np.linspace(0, max(errors_bestref), int(max(errors_bestref)/5)),
         histtype='bar', ec='black', color="tab:blue")
plt.title('Error Histogram - Best Reference')
plt.xlabel('Angle Error ($^o$)')
plt.ylabel('Frequency')
plt.locator_params(axis='y', integer=True)
plt.tight_layout ()
plt.savefig ('best_params_error_bestref.eps', format = 'eps')
plt.savefig ('best_params_error_bestref.png', format = 'png' , dpi=400)

plt.figure()
# plt.grid()
plt.hist(times_bestref, bins=np.linspace(0, max(times_bestref), int(max(times_bestref)/5)),
         histtype='bar', ec='black', color="tab:blue")
plt.title('Processing Time Histogram - Best Reference')
plt.xlabel('Processing Time (ms)')
plt.ylabel('Frequency')
plt.locator_params(axis='y', integer=True)
plt.tight_layout ()
plt.savefig ('best_params_time_bestref.eps', format = 'eps')
plt.savefig ('best_params_time_bestref.png', format = 'png' , dpi=400)

plt.figure()
# plt.grid()
plt.hist(errors_irl, bins=np.linspace(0, max(errors_irl), int(max(errors_irl)/5)),
         histtype='bar', ec='black', color="tab:orange")
plt.title('Error Histogram - Recover Pose - Real Field')
plt.xlabel('Angle Error ($^o$)')
plt.ylabel('Frequency')
plt.locator_params(axis='y', integer=True)
plt.tight_layout ()
plt.savefig ('best_params_error_irl.eps', format = 'eps')
plt.savefig ('best_params_error_irl.png', format = 'png' , dpi=400)
