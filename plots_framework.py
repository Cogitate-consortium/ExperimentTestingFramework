import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set Helvetica as the default font:
plt.rcParams["font.family"] = "sans-serif"
plt.rc('font', size=22)  # controls default text sizes
plt.rc('axes', titlesize=22)  # fontsize of the axes title
plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)  # fontsize of the tick labels
plt.rc('ytick', labelsize=22)  # fontsize of the tick labels
plt.rc('legend', fontsize=22)  # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the fi
figsize = [183/25.4, 108/25.4]
# Create the time vector:
tmin = 0
tmax = 2000
sfreq = 500
times = np.linspace(tmin, tmax, int((tmax - tmin) * sfreq))
n_events = 500

# Generate events that can have one of three different intervals:
intervals = [0.5, 1, 1.5]
event_ts = np.zeros(n_events)
event_ts[0] = np.random.choice(intervals)
for i in range(n_events - 1):
    event_ts[i + 1] = np.random.choice(intervals) + event_ts[i]

# ================================================================
# Generate a photodiode signal:
event_ts_photodiode = event_ts + np.random.normal(0, 0.0025, n_events)

# Convert the photodiode event to stick predictor:
event_photodiode = np.zeros(len(times))
for event in event_ts_photodiode:
    event_photodiode[np.where(times < event)[0][-1]] = 1

# Convolve a box car with the stick predictor:
box_car = np.ones(int(0.1 * sfreq))
event_photodiode = np.convolve(event_photodiode, box_car, mode="same")

# Add some noise:
event_photodiode = event_photodiode + np.random.normal(0, 0.05, len(event_photodiode))

# ================================================================
# Perform framework steps on the photodiode signal:
# 1. Plot the signal:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10 sec of data:
ax.plot(times[:int(10 * sfreq)], event_photodiode[:int(10 * sfreq)])
# Add the threshold:
ax.hlines(0.5, xmin=0, xmax=10, color="red", label="Threshold")
ax.legend()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Photodiode signal (a.u.)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("raw_photodiode_signal.svg")
# Show the figure:
plt.show()

# 2. Binarize the signal:
threshold = 0.5
event_photodiode_bin = event_photodiode > threshold
# Plot the binarized signal:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10sec of data:
ax.plot(times[:int(10 * sfreq)], event_photodiode_bin[:int(10 * sfreq)])
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Photodiode signal (a.u.)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("binarized_photodiode_signal.svg")
# Show the figure:
plt.show()

# 3. Compute the discrete difference to find the events onsets:
event_photodiode_bin_diff = np.diff(event_photodiode_bin.astype(int))
# Plot the signal:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10sec of data:
ax.plot(times[:int(10 * sfreq)], event_photodiode_bin_diff[:int(10 * sfreq)])
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Photodiode signal (a.u.)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("diff_photodiode_signal.svg")
# Show the figure:
plt.show()

# 4. Find the events onsets:
event_photodiode_onsets = times[np.where(event_photodiode_bin_diff == 1)]
# Plot the signal:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10sec of data:
ax.plot(times[:int(10 * sfreq)], event_photodiode_bin_diff[:int(10 * sfreq)])
ax.vlines(event_photodiode_onsets[np.where(event_photodiode_onsets <= 10)], ymin=0, ymax=1, color="red", label="Onsets")
ax.legend()
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Photodiode signal (a.u.)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("onsets_photodiode_signal.svg")
# Show the figure:
plt.show()

# 5. Compute the intervals between events:
event_photodiode_intervals = np.diff(event_photodiode_onsets)
# Plot the signal:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10 intervals:
ax.plot(event_photodiode_intervals[:20])
ax.set_xlabel("Event number")
ax.set_ylabel("Interval (sec)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("intervals_photodiode_signal.svg")
# Show the figure:
plt.show()

# ================================================================
# Generate the log file data:
event_ts_log = event_ts + np.random.normal(0, 0.0025, n_events)

# Compute the interval between events:
event_log_intervals = np.diff(event_ts_log)
# Plot the intervals:
fig, ax = plt.subplots(figsize=figsize)
# Plot only the first 10 intervals:
ax.plot(event_log_intervals[:20], color="orange")
ax.set_xlabel("Event number")
ax.set_ylabel("Interval (sec)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Save the figure to a svg:
fig.savefig("intervals_log.svg")
# Show the figure:
plt.show()

# Plot the two on top of each other:
fig, ax = plt.subplots(figsize=figsize)
ax.plot(event_photodiode_intervals, label="Photodiode")
ax.plot(event_log_intervals, label="Log")
ax.set_xlabel("Event number")
ax.set_ylabel("Interval (sec)")
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()
# Save the figure to a svg:
plt.show()

# ================================================================
# Compare the two:
# Compute the difference between the two:
event_diff = event_photodiode_intervals - event_log_intervals

# Plot the difference:
fig, ax = plt.subplots(figsize=figsize)
ax.hist(event_diff, bins=20)
ax.set_xlabel("Difference (sec)")
ax.set_ylabel("Count")
# Add as text the mean and sd:
ax.text(0.8, 0.9, "Mean: {:.3f}\nSD: {:.3f}".format(np.mean(event_diff), np.std(event_diff)),
        transform=ax.transAxes)
# Remove the top and right spines:
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# plt.tight_layout()
# Save the figure to a svg:
fig.savefig("diff_intervals.svg")
# Show the figure:
plt.show()
