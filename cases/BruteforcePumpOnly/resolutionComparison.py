import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

res  = [0.2, 0.16, 0.125, 0.1, 0.064]
eta  = [0.36368, 0.4825, 0.55825, 0.58165, 0.60886]
mins = [3, 6, 13, 24, 115]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.scatter(res, eta, color="black")
ax1.set_ylabel("Eta")

ax2.scatter(res, mins, color="grey")
ax2.set_ylabel("Mins")
ax2.set_xlabel("Resolution")
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
