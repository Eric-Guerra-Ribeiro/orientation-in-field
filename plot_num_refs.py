import matplotlib.pyplot as plt
import numpy as np

num_refs_recoverpose = np.loadtxt("./results/num_refs/num_refs_recoverpose.txt")
times_mean_recoverpose = np.loadtxt("./results/num_refs/times_mean_recoverpose.txt")
times_std_recoverpose = np.loadtxt("./results/num_refs/times_std_recoverpose.txt")
errors_mean_recoverpose = np.loadtxt("./results/num_refs/errors_mean_recoverpose.txt")
errors_std_recoverpose = np.loadtxt("./results/num_refs/errors_std_recoverpose.txt")

num_refs_bestref = np.loadtxt("./results/num_refs/num_refs_bestref.txt")
times_mean_bestref = np.loadtxt("./results/num_refs/times_mean_bestref.txt")
times_std_bestref = np.loadtxt("./results/num_refs/times_std_bestref.txt")
errors_mean_bestref = np.loadtxt("./results/num_refs/errors_mean_bestref.txt")
errors_std_bestref = np.loadtxt("./results/num_refs/errors_std_bestref.txt")

plt.rc('text' , usetex = True)
plt.rcParams.update({ 'font.family': 'serif' })
plt.rcParams.update({'font.size': 18})

plt.figure()
plt.grid()
plt.errorbar(
    num_refs_bestref, times_mean_bestref, capsize=3,
    yerr=times_std_bestref, elinewidth=1, fmt='o'
)
plt.errorbar(
    num_refs_recoverpose, times_mean_recoverpose, capsize=3,
    yerr=times_std_recoverpose, elinewidth=1, fmt='o'
)
plt.xlim(0, 25)
plt.ylim(0, 80)
plt.xlabel('Number of References')
plt.ylabel('Processing Time ($ms$)')
plt.legend (['Best Reference' , 'Recover Pose'])
plt.tight_layout ()
plt.savefig ('num_refs_time.eps', format = 'eps')
plt.savefig ('num_refs_time.png', format = 'png' , dpi=400)

plt.figure()
plt.grid()
plt.errorbar(
    num_refs_bestref, errors_mean_bestref, capsize=3,
    yerr=errors_std_bestref, elinewidth=1, fmt='o'
)
plt.errorbar(
    num_refs_recoverpose, errors_mean_recoverpose, capsize=3,
    yerr=errors_std_recoverpose, elinewidth=1, fmt='o'
)
plt.xlim(0, 25)
plt.ylim(0, 140)
plt.xlabel('Number of References')
plt.ylabel('Angle error ($º$)')
plt.legend (['Best Reference' , 'Recover Pose'])
plt.tight_layout ()
plt.savefig ('num_refs_error.eps', format = 'eps')
plt.savefig ('num_refs_error.png', format = 'png' , dpi=400)

