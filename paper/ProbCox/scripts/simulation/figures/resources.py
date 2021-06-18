# Modules
# =======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt


# Plot Settings
# =======================================================================================================================

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Data - #extracted from the logs
# =======================================================================================================================
rsurvival_t = [2, 30, 241, 2309, 40444]
rsurvival_m = [250, 544, 1697, 6774, 21507]
#rsurvival_c = [91, 85, 83, 85]

rglmnet_t = [57, 71, 405, 793, 3904]
rglmnet_m = [333, 669, 2145, 8573, 28180]

probcox_t = [126, 158, 279, 374, 620]
probcox_m = [258, 318, 672, 2120, 7631]
#probcox_c = [88, 77, 76, 81]

probcox2_t = [477, 539, 636, 827, 1219]
probcox2_m = [247, 249, 253, 263, 272]


# Plot
# =======================================================================================================================

fig, ax = plt.subplots(4, 1, figsize=(8.27*0.5, 11.69/2), dpi=300, sharex=False, gridspec_kw={'height_ratios': [0.5, 3, 3, 1]})
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.075)
ax[3].set_yticklabels([])
ax[3].set_yticks([])
ax[3].set_xticks([])
ax[3].set_xticklabels([])
ax[2].set_xticklabels([])
ax[1].set_xticklabels([])
ax[3].spines['left'].set_visible(False)
ax[3].spines['bottom'].set_visible(False)
ax[3].text(x=-.25, y=0 , s=r'\noindent Individuals:' "\n" r'Observations:' "\n" r'Events:' "\n" r'Covariates:')

ax[3].text(x=.02, y=0 , s=r'\noindent 1000' "\n" r'7428' "\n" r'230' "\n" r'200')
ax[3].text(x=.24, y=0 , s=r'\noindent 2000' "\n" r'15600' "\n" r'356' "\n" r'400')
ax[3].text(x=.47, y=0 , s=r'\noindent 4000' "\n" r'30365' "\n" r'940' "\n" r'800')
ax[3].text(x=.69, y=0 , s=r'\noindent 8000' "\n" r'61710' "\n" r'1615' "\n" r'1600')
ax[3].text(x=.9, y=0 , s=r'\noindent 16000' "\n" r'119071' "\n" r'4034' "\n" r'3200')

ax[1].set_ylabel(r'Compute time in seconds')
ax[2].set_ylabel(r'Memory in Mb')
ax[1].plot([0.2, 0.4, 0.6, 0.8, 1], rsurvival_t, label=r'R-Survival', color='.65')
ax[1].plot([0.2, 0.4, 0.6, 0.8, 1], rglmnet_t, label=r'R-glmnet', color='.65', ls=':')
ax[1].plot([0.2, 0.4, 0.6, 0.8, 1], probcox_t, label=r'ProbCox', color='.2')
ax[1].plot([0.2, 0.4, 0.6, 0.8, 1], probcox2_t, label=r'ProbCox - Hard drive', color='.2', ls='--')
ax[2].plot([0.2, 0.4, 0.6, 0.8, 1], rsurvival_m, color='.65')
ax[2].plot([0.2, 0.4, 0.6, 0.8, 1], rglmnet_m, color='.65', ls=':')
ax[2].plot([0.2, 0.4, 0.6, 0.8, 1], probcox_m, color='.2')
ax[2].plot([0.2, 0.4, 0.6, 0.8, 1], probcox2_m, color='.2', ls='--')
ax[1].legend(frameon=False, loc='center left', prop={'size': 8})

ax[1].set_ylim([0, 10000])
ax[1].set_xticks([0.15, 0.2, 0.4, 0.6, 0.8, 1])
ax[2].set_xticks([0.15, 0.2, 0.4, 0.6, 0.8, 1])

ax[0].plot([0.68, 1], [30000, rsurvival_t[-1]], color='.65', linewidth=1.2)
ax[0].set_xticks([0.15, 0.2, 0.4, 0.6, 0.8, 1])
ax[0].xaxis.set_ticks_position('none')

ax[0].spines['bottom'].set_visible(False)
ax[0].set_xticklabels([])
ax[0].set_ylim([35000, 40000])
ax[0].set_yticks([40000])

ax[1].set_yticks([0, 4000, 8000])

d = .015
kwargs = dict(transform=ax[1].transAxes, color='k', clip_on=False)
kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
ax[1].plot((-d, +d), (0.988 - d, 0.988 + d), **kwargs)  # bottom-left diagonal

ax[1].plot((-d, +d), (1.058 - d, 1.058 + d), **kwargs)  # bottom-left diagonal

plt.savefig('/Users/alexwjung/projects/ProbCox/paper/ProbCox/out/simulation/figures/resources.eps', bbox_inches='tight', dpi=300)
