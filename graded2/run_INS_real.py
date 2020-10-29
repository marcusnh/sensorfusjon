# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try: # see if tqdm is available, otherwise define it as a dummy
    try: # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
        __IPYTHON__
        from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
except Exception as e:
    print(e)
    print(
        "install tqdm (conda install tqdm, or pip install tqdm) to get nice progress bars. "
    )

    def tqdm(iterable, *args, **kwargs):
        return iterable

from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

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
filename_to_load = "task_real.mat"
loaded_data = scipy.io.loadmat(filename_to_load)

do_corrections = True # TODO: set to false for the last task
if do_corrections:
    S_a = loaded_data['S_a']
    S_g = loaded_data['S_g']
else:
    # Only accounts for basic mounting directions
    S_a = S_g = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T
accuracy_GNSS = loaded_data['GNSSaccuracy'].ravel()

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration)
gnss_steps = len(z_GNSS)

# %% Measurement noise
# Continous noise
rate_std =0.15*np.pi/180*(1/np.sqrt(3600)) # last value is tuning #from degree/sqrt(hour) to rad/sqrt(s)
acc_std =  0.06/(np.sqrt(3600)) *0.99 #from  m/(s*sqrt(h)) to m/(s*sqrt(s))


# Bias values
cont_rate_bias_driving_noise_std = 0.5*np.pi/180*(1/np.sqrt(3600))*1.1 #from degree/sqrt(hour) to rad/s

cont_acc_bias_driving_noise_std = 0.05*9.81/1000*0.99

# Position and velocity measurement
p_acc = 1e-16 # TODO # Bias Instability

p_gyro = 1e-16  # TODO # Bias Instability

# %% Estimator
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a = S_a, # set the accelerometer correction matrix
    S_g = S_g, # set the gyro correction matrix,
    debug=True # False to avoid expensive debug checks
)


# %% Allocate
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

NIS = np.zeros(gnss_steps)

# %% Initialise
x_pred[0, POS_IDX] = np.array([0, 0, -5]) # starting 5 metres above ground
x_pred[0, VEL_IDX] = np.array([20, 0, 0]) # starting at 20 m/s due north
x_pred[0, ATT_IDX] = np.array([
    np.cos(45 * np.pi / 180),
    0, 0,
    np.sin(45 * np.pi / 180)
])  # nose to east, right to south and belly down.

P_pred[0][POS_IDX**2] = 10**2 * np.eye(3)
P_pred[0][VEL_IDX**2] = 3**2 * np.eye(3)
P_pred[0][ERR_ATT_IDX**2] = (np.pi/30)**2 * np.eye(3) # error rotation vector (not quat)
P_pred[0][ERR_ACC_BIAS_IDX**2] = 0.05**2 * np.eye(3)
P_pred[0][ERR_GYRO_BIAS_IDX**2] = 2*(1e-3)**2 * np.eye(3)


p_std = np.array([0.4, 0.4, 0.4]) # Measurement noise
R_GNSS_const = np.diag(p_std ** 2)

# %% Run estimation

N = round(len(z_acceleration)/9)-100 #steps
start = 0

GNSSk = 0
for x in range(start):
    if timeIMU[x] >= timeGNSS[GNSSk]:
        GNSSk += 1
    x_est[x] = x_pred[x]
    P_est[x] = P_pred[x]
    x_pred[x+1] = x_est[x]
    P_pred[x+1] = P_pred[x]

start_GNSSk = GNSSk  # Saving the variable for calculating NIS from start.
for k in tqdm(range(start, N)):
    if timeIMU[k] >= timeGNSS[GNSSk]:
        R_GNSS = 0.255*np.diag(p_std*accuracy_GNSS[GNSSk]) ** 2  # TODO: Current GNSS covariance
        #R_GNSS = np.diag(p_std*accuracy_GNSS[GNSSk]/np.mean(accuracy_GNSS[:GNSSk]))


        NIS[GNSSk] = eskf.NIS_GNSS_position(
            x_pred[k, :], P_pred[k, :, :], z_GNSS[GNSSk, :], R_GNSS, lever_arm=lever_arm
        )  # TODO

        NIS[GNSSk] = eskf.NIS_GNSS_position(
            x_pred[k, :], P_pred[k, :, :], z_GNSS[GNSSk, :], R_GNSS, lever_arm=lever_arm
        )  # TODO

        x_est[k], P_est[k] = eskf.update_GNSS_position(
            x_pred[k, :], P_pred[k, :, :], z_GNSS[GNSSk, :], R_GNSS, lever_arm
        )  # TODO
        if eskf.debug:
            assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"

        GNSSk += 1
    else:
        # no updates, so estimate = prediction
        x_est[k] = x_pred[k]  # TODO
        P_est[k] = P_pred[k]  # TODO

    if k < N - 1:
        x_pred[k + 1], P_pred[k + 1] = eskf.predict(
            x_est[k, :], P_est[k, :, :], z_acceleration[k+1, :], z_gyroscope[k+1, :], dt
        )  # TODO

    if eskf.debug:
        assert np.all(np.isfinite(P_pred[k])), f"Not finite P_pred at index {k + 1}"


# %% Plots

fig1 = plt.figure(1)
ax = plt.axes(projection='3d')

ax.plot3D(x_est[0:N, 1], x_est[0:N, 0], -x_est[0:N, 2], color="b", label= "Estimated position")
ax.plot3D(z_GNSS[0:GNSSk, 1], z_GNSS[0:GNSSk, 0], -z_GNSS[0:GNSSk, 2], color="red", label= "GNSS measurment", linestyle="dashed")
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_zlabel("Altitude [m]")
ax.legend()

plt.grid()

fig5 = plt.figure()
ax5 = plt.axes(projection=None)

ax5.plot(x_est[0:N, 1], x_est[0:N, 0], color="b", label= "Estimated position")
ax5.plot(z_GNSS[0:GNSSk, 1], z_GNSS[0:GNSSk, 0], color="red", label= "GNSS measurment", linestyle="dashed")
ax5.set_xlabel('East [m]')
ax5.set_ylabel('North [m]')
ax5.legend()

plt.grid()

# state estimation
t = np.linspace(0, dt*(N-1), N)
eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])

fig2, axs2 = plt.subplots(5, 1)

axs2[0].plot(t, x_est[0:N, POS_IDX])
axs2[0].set(ylabel='NED position [m]')
axs2[0].legend(['North', 'East', 'Down'],loc='upper right')
plt.grid()

axs2[1].plot(t, x_est[0:N, VEL_IDX])
axs2[1].set(ylabel='Velocities [m/s]')
axs2[1].legend(['North', 'East', 'Down'],loc='upper right')
plt.grid()

axs2[2].plot(t, eul[0:N] * 180 / np.pi)
axs2[2].set(ylabel='Euler angles [deg]')
axs2[2].legend([r"$\phi$", r"$\theta$", r"$\psi$"],loc='upper right')
plt.grid()

axs2[3].plot(t, x_est[0:N, ACC_BIAS_IDX])
axs2[3].set(ylabel='Accl bias [m/s^2]')
axs2[3].legend(['x', 'y', 'z'],loc='upper right')
plt.grid()

axs2[4].plot(t, x_est[0:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
axs2[4].set(ylabel='Gyro bias [deg/h]')
axs2[4].legend(['x', 'y', 'z'],loc='upper right')
plt.grid()

fig2.suptitle('States estimates')

# %% Consistency
confprob = 0.95
CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))

CI3K = np.array(scipy.stats.chi2.interval(confprob, 3 * N)) / N
ANIS = np.mean(NIS[start_GNSSk:GNSSk])
print(f"ANIS = {ANIS:.2f} with CI = [{CI3K[0]:.2f}, {CI3K[1]:.2f}]")

fig3 = plt.figure()

plt.plot(NIS[start_GNSSk:GNSSk])
plt.plot(np.array([0, (N-start)-1]) * dt, (CI3@np.ones((1, 2))).T,color='r',linestyle= 'dashed')
insideCI = np.mean((CI3[0] <= NIS[start_GNSSk:GNSSk]) * (NIS[start_GNSSk:GNSSk] <= CI3[1]))
plt.title(f'NIS ({100 *  insideCI:.1f} inside {100 * confprob} confidence interval)')
plt.grid()

# %% box plots
fig4 = plt.figure()

gauss_compare = np.sum(np.random.randn(3, GNSSk)**2, axis=0)
plt.boxplot([NIS[0:GNSSk], gauss_compare], notch=True)
plt.legend(('NIS', 'gauss'))
plt.grid()

plt.show()



# %%
