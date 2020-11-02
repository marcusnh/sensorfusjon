# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import dynamicmodels
import measurementmodels
from gaussparams import GaussParams
from mixturedata import MixtureParameters
import ekf
import imm
import pda

from typing import List
import estimationstatistics as estats

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T #4X4
Z = [zk.T for zk in loaded_data["Z"].ravel()]
#alternative 1:
#Ts = Ts.mean()
#alternative 2:

Ts = np.insert(Ts, 0, Ts.mean(), axis=0)



# plot measurements close to the trajectory

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="r")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
#ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)

# %% setup and track

# initialize sensor
sigma_z = 25
clutter_intensity = 1e-5*10*2
PD = 0.6
gate_size = 3.5



# dynmaic models
sigma_a_CV = 2
sigma_a_CV_high = 3
sigma_a_CT = 1.3 # change to 1.3
sigma_omega = 0.005*np.pi # change to 0.001


# markov chain
PI11 = 0.9
PI22 = 0.9
PI33 = 0.9

p10 = 0.9  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
PI_3f = np.array([[PI11,(1 - PI11)/2,(1 - PI11)/2],[(1 - PI22)/2, PI22 , (1 - PI22)/2],[(1 - PI33)*3/4,(1 - PI33)/4, PI33]])
print(PI_3f)
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"
assert np.allclose(np.sum(PI_3f, axis=1), 1), "rows of PI must sum to 1"

#starting positions:
mean_init =  [7000, 3600 ,0,0,0.01*np.pi]# Xgt[0,:]
cov_init = np.diag([50, 25, 3, 0.01*np.pi, 0.005*np.pi]) ** 2 # z is a 4X4 matrix not 5x5 as in previous task?
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_prob_init_3f = np.array([p10,(1-p10)/2,(1-p10)/2])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 2)
init_imm_state_3f = MixtureParameters(mode_prob_init_3f, [mode_states_init] * 3) # why *3?

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"
assert np.allclose(
    np.sum(mode_prob_init_3f), 1
), "initial mode probabilities must sum to 1"

# create our models:
# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)

dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=5))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter_CVCT = imm.IMM(ekf_filters, PI)
ekf_filters.append(ekf.EKF(dynamic_models[2], measurement_model))
imm_filter_CVCT_CVH = imm.IMM(ekf_filters,PI_3f)

tracker = pda.PDA(imm_filter_CVCT, clutter_intensity, PD, gate_size)
tracker_3f = pda.PDA(imm_filter_CVCT_CVH, clutter_intensity, PD, gate_size)


#model paramteres:
NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

NEES_3f = np.zeros(K)
NEESpos_3f = np.zeros(K)
NEESvel_3f = np.zeros(K)

tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []

tracker_update_3f = init_imm_state_3f
tracker_update_list_3f = []
tracker_predict_list_3f = []
tracker_estimate_list_3f = []

# estimate

for k, (Zk, x_true_k, Ts) in enumerate(zip(Z, Xgt, Ts)):
    tracker_predict = tracker.predict(tracker_update, Ts)
    tracker_predict_3f = tracker_3f.predict(tracker_update_3f, Ts)
    tracker_update = tracker.update(Zk, tracker_predict)
    tracker_update_3f = tracker_3f.update(Zk, tracker_predict_3f)

    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)
    tracker_estimate_3f = tracker_3f.estimate(tracker_update_3f)

    NEES[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
    NEESpos[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
    NEESvel[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

    NEES_3f[k] = estats.NEES(*tracker_estimate_3f, x_true_k, idxs=np.arange(4))
    NEESpos_3f[k] = estats.NEES(*tracker_estimate_3f, x_true_k, idxs=np.arange(2))
    NEESvel_3f[k] = estats.NEES(*tracker_estimate_3f, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)
    
    tracker_predict_list_3f.append(tracker_predict_3f)
    tracker_update_list_3f.append(tracker_update_3f)
    tracker_estimate_list_3f.append(tracker_estimate_3f)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])   

x_hat_3f = np.array([est.mean for est in tracker_estimate_list_3f])
prob_hat_3f = np.array([upd.weights for upd in tracker_update_list_3f])   


# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()

#for CT-CV-CV_H:
# calculate a performance metrics
poserr_3f = np.linalg.norm(x_hat_3f[:, :2] - Xgt[:, :2], axis=1)
velerr_3f = np.linalg.norm(x_hat_3f[:, 2:4] - Xgt[:, 2:4], axis=1)
posRMSE_3f = np.sqrt(
    np.mean(poserr_3f ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE_3f = np.sqrt(np.mean(velerr_3f ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation_3f = poserr_3f.max()
peak_vel_deviation_3f = velerr_3f.max()


# consistency
confprob = 0.95
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

ANEESpos_3f = np.mean(NEESpos_3f)
ANEESvel_3f = np.mean(NEESvel_3f)
ANEES_3f = np.mean(NEES_3f)


# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$",color='r',)
axs3[0].plot(*x_hat_3f.T[:2], label=r"$\hat x$ With 3 filter",color='y',)
axs3[0].plot(*Xgt.T[:2], label="$x$",color='k')
axs3[0].set_title(
    f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f}\n\
    RMSE_3f(pos, vel) = ({posRMSE_3f:.3f}, {velRMSE_3f:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation_3f:.3f}, {peak_vel_deviation_3f:.3f}))"
)
axs3[0].axis("equal")
axs3[0].legend()
# probabilities
axs3[1].plot(np.arange(K) * Ts, prob_hat_3f[:,0],color='g',label='CV')
axs3[1].plot(np.arange(K) * Ts, prob_hat_3f[:,1],color='b',label='CT')
axs3[1].plot(np.arange(K) * Ts, prob_hat_3f[:,2],color='m',label='CV_high')
axs3[1].legend()
axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time")
axs3[1].set_title("CV mode sigma:{}\n CV_high mode sigma:{} \n CT mode sigma:{}".format(sigma_a_CV,sigma_a_CV_high, sigma_a_CT))

# NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
fig4.suptitle('NEES performance with CV and CT', fontsize=12)
axs4[0].plot(np.arange(K) * Ts, NEESpos)
axs4[0].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(np.arange(K) * Ts, NEESvel)
axs4[1].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel<= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(np.arange(K) * Ts, NEES)
axs4[2].plot([0, (K - 1) * Ts], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES) * (NEES <= CI4[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# NEES for CT-CV-CV_H:
fig5, axs5 = plt.subplots(3, sharex=True, num=5, clear=True)
fig5.suptitle('NEES performance with CV, CV_high and CT', fontsize=12)
axs5[0].plot(np.arange(K) * Ts, NEESpos_3f)
axs5[0].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs5[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos_3f) * (NEESpos_3f <= CI2[1]))
axs5[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs5[1].plot(np.arange(K) * Ts, NEESvel_3f)
axs5[1].plot([0, (K - 1) * Ts], np.repeat(CI2[None], 2, 0), "--r")
axs5[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel_3f) * (NEESvel_3f <= CI2[1]))
axs5[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs5[2].plot(np.arange(K) * Ts, NEES_3f)
axs5[2].plot([0, (K - 1) * Ts], np.repeat(CI4[None], 2, 0), "--r")
axs5[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES_3f) * (NEES_3f <= CI4[1]))
axs5[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")
print("\nNEES for CT-CV-CV_H:")
print(f"ANEESpos = {ANEESpos_3f:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel_3f:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES_3f:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")
# errors
fig6, axs6 = plt.subplots(2, 2, num=6, clear=True)
axs6[0][0].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs6[0][0].set_ylabel("position error")
axs6[0][0].set_title("For CV and CT model")

axs6[1][0].plot(np.arange(K) * Ts, np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs6[1][0].set_ylabel("velocity error")

axs6[0][1].plot(np.arange(K) * Ts, np.linalg.norm(x_hat_3f[:, :2] - Xgt[:, :2], axis=1))
axs6[0][1].set_ylabel("position error")
axs6[0][1].set_title("For CV, CV_high and CT model")

axs6[1][1].plot(np.arange(K) * Ts, np.linalg.norm(x_hat_3f[:, 2:4] - Xgt[:, 2:4], axis=1))
axs6[1][1].set_ylabel("velocity error")


plt.show()


