# %% Imports
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

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



from EKFSLAM import EKFSLAM
from plotting import ellipse

# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")

## NB: this is a MATLAB cell, so needs to "double index" to get out the measurements of a time step k:
#
# ex:
#
# z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
#
##
z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T

K = len(z)
print(K)
M = len(landmarks)
print(M)

# %% Initilize
Q = np.diag([4.5e-1,4.5e-1,1.2e-2])**2 # TODO
R =  np.diag([4.5e-2, 2.5e-2])**2 # TODO

doAsso = True

JCBBalphas = np.array(
    # TODO,
    [1e-6, 1e-6]
)  # first is for joint compatibility, second is individual
# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.
# %% Run simulation
N = round(K/1)

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

# allocate
eta_pred: List[Optional[np.ndarray]] = [None] * N
P_pred: List[Optional[np.ndarray]] = [None] * N
eta_hat: List[Optional[np.ndarray]] = [None] * N
P_hat: List[Optional[np.ndarray]] = [None] * N
a: List[Optional[np.ndarray]] = [None] * N
NIS = np.zeros(N)
NISnorm = np.zeros(N)
CI = np.zeros((N, 2))
CInorm = np.zeros((N, 2))
NEESes = np.zeros((N, 3))

# For consistency testing
alpha = 1-0.05

# init
eta_pred[0] = poseGT[0]  # we start at the correct position for reference
P_pred[0] = np.zeros((3, 3))  # we also say that we are 100% sure about that

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = False
if doAssoPlot:
    figAsso, axAsso = plt.subplots(num=1, clear=True)



print("starting sim (" + str(N) + " iterations)")
pos_hat: List[Optional[np.ndarray]] = [None] * K
pos_hat[0] = poseGT[0][:3]

for k, z_k in tqdm(enumerate(z[:N])):

    eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(eta_pred[k], P_pred[k], z[k]) # TODO update



    if k < N - 1:
        eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k],P_hat[k], odometry[k, :]) # TODO predict

    assert (
        eta_hat[k].shape[0] == P_hat[k].shape[0]
    ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI[k] = chi2.interval(alpha, 2 * num_asso)

    if num_asso > 0:
        NISnorm[k] = NIS[k] / (2 * num_asso)
        CInorm[k] = CI[k] / (2 * num_asso)
    else:
        NISnorm[k] = 1
        CInorm[k].fill(1)

    NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3,:3], poseGT[k,:]) # TODO, use provided function slam.NEESes




print("sim complete")
pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

fig2, ax2 = plt.subplots(num=2, clear=True)
# landmarks
ax2.scatter(*landmarks.T, c="r", marker="^")
ax2.scatter(*lmk_est_final.T, c="y", marker=".")
# Draw covariance ellipsis of measurements
for l, lmk_l in enumerate(lmk_est_final):
    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
    rI = P_hat[N - 1][idxs, idxs]
    el = ellipse(lmk_l, rI, 5, 200)
    ax2.plot(*el.T, "y")

ax2.plot(*poseGT.T[:2], c="g", label="Ground truth", linestyle='--')
ax2.plot(*pose_est.T[:2], c="b", label="Estimated position")
ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="b")
ax2.set(title="Trajectory", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
ax2.axis("equal")
ax2.legend()
ax2.grid()

# %% Consistency

# NIS
insideCI = (CInorm[:N,0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N,1])
ANIS = np.mean(NISnorm[:N])
CI_NIS =np.array(chi2.interval(alpha, 3*N)) / N
print(f"CI ANIS all: {CI_NIS}")
print(f"ANIS all: {ANIS}")

fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(CInorm[:N,0], '--', color='r')
ax3.plot(CInorm[:N,1], '--', color='r')
ax3.plot(NISnorm[:N], lw=0.5)

ax3.set_title(f'NIS, {round(insideCI.mean()*100,2)}% inside CI')

# NEES

fig4, ax4 = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), num=4, clear=True, sharex=True)
tags = ['all', 'pos', 'heading']
dfs = [3, 2, 1]

for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
    CI_NEES = chi2.interval(alpha, df)
    ax.plot(np.full(N, CI_NEES[0]), '--', color='r')
    ax.plot(np.full(N, CI_NEES[1]), '--', color='r')
    ax.plot(NEES[:N], lw=0.5)
    insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
    ax.set_title(f'NEES {tag}: {round(insideCI.mean()*100,2)}% inside CI')

    CI_ANEES = np.array(chi2.interval(alpha, df*N)) / N
    print(f"CI ANEES {tag}: {CI_ANEES}")
    print(f"ANEES {tag}: {NEES.mean()}")

fig4.tight_layout()

# %% RMSE

ylabels = ['m', 'deg']
scalings = np.array([1, 180/np.pi])

fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), num=5, clear=True, sharex=True)

pos_err = np.linalg.norm(pose_est[:N,:2] - poseGT[:N,:2], axis=1)
heading_err = np.abs(utils.wrapToPi(pose_est[:N,2] - poseGT[:N,2]))

errs = np.vstack((pos_err, heading_err))

for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
    ax.plot(err*scaling)
    ax.set_title(f"{tag}: RMSE {round(np.sqrt((err**2).mean())*scaling,2)} {ylabel}")
    ax.set_ylabel(f"[{ylabel}]")
    ax.grid()

fig5.tight_layout()

# %% Movie time

if playMovie:
    try:
        print("recording movie...")

        from celluloid import Camera

        pauseTime = 0.05
        fig_movie, ax_movie = plt.subplots(num=6, clear=True)

        camera = Camera(fig_movie)

        ax_movie.grid()
        ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
        camera.snap()

        for k in tqdm(range(N)):
            ax_movie.scatter(*landmarks.T, c="r", marker="^")
            ax_movie.plot(*poseGT[:k, :2].T, "r-")
            ax_movie.plot(*pose_est[:k, :2].T, "g-")
            ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

            if k > 0:
                el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                ax_movie.plot(*el.T, "g")

            numLmk = lmk_est[k].shape[0]
            for l, lmk_l in enumerate(lmk_est[k]):
                idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                rI = P_hat[k][idxs, idxs]
                el = ellipse(lmk_l, rI, 5, 200)
                ax_movie.plot(*el.T, "b")

            camera.snap()
        animation = camera.animate(interval=100, blit=True, repeat=False)
        print("playing movie")

    except ImportError:
        print(
            "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
        )

plt.show()
# %%
