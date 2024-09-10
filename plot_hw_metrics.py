import matplotlib.pyplot as plt
import numpy as np

token_keep_rate_eventful = [0.58, 0.435, 0.29, 0.22, 0.145, 0.0725]
token_count_eventful = [1024, 768, 512, 384, 256, 128]
map_eventful = [82.23, 82.21, 81.84, 81.43, 80.16, 75.19]
gflops_eventful = [115.1, 87.9, 60.7, 47.1, 33.5, 19.9]
latency_eventful = [51.68, 47.66, 43.78, 42.28, 41.696, 38.04]
memory_eventful = [2126.06, 2123.57, 2121.59, 2121.54, 2119.14, 2115.28]

token_keep_rate_ours = [0.54, 0.47, 0.43, 0.402, 0.36, 0.33, 0.27, 0.22, 0.20]
token_count_ours = [952, 829, 758, 709, 635, 582, 476, 388, 352]
map_ours = [82.11, 81.9, 81.52, 81.81, 81.33, 81.78, 81.68, 81.06, 80.51]
gflops_ours = [106.43, 94.98, 89.19, 85.01, 78.47, 75.29, 65.49, 57.56, 54.84]
latency_ours = [42.15, 40.89, 39.89, 39.61, 38, 38.95, 37.63, 37.03, 35.71]
memory_ours = [928, 928, 928.36, 928, 928, 928.74, 928, 928.75, 927.8]

map_stgt = [82.28, 81.95, 80.45, 78.71, 75.57, 68.13]
gflops_stgt = [111.9, 90.2, 68.5, 57.7, 46.8, 36]
latency_stgt = [44.62, 41.81, 39.17, 37.66, 36.38, 35.17]
memory_stgt = [1249.45, 1249.45, 1246.96, 1245.39, 1242.9, 1238.68]
# Plot with multiple subplots
fig, axs = plt.subplots(1, 4, figsize=(24, 5))
# CIFAR-10
axs[0].plot(token_keep_rate_eventful, map_eventful, c='navy', label='Eventful', marker='*', markersize=20, linewidth=3.5)
axs[0].plot(token_keep_rate_eventful, map_stgt, c='green', label='STGT', marker='*', markersize=20, linewidth=3.5)
axs[0].plot(token_keep_rate_ours, map_ours, c='orange', label='Ours', marker='*', markersize=20, linewidth=3.5)
axs[0].set_xlabel('Patch Keep Rate', size=20)
axs[0].set_ylabel('mAP-50', size=20)
axs[0].tick_params(axis='x', labelsize=18)
axs[0].tick_params(axis='y', labelsize=18)
# axs[0].set_title('map-50', size=20, y=1.3)
axs[0].grid(True, which='both', linestyle='--', linewidth=1.5)
# axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
#           ncol=2, prop ={'size':20})
axs[1].plot(token_keep_rate_eventful, gflops_eventful, c='navy', label='Eventful', marker='*', markersize=20, linewidth=3.5)
axs[1].plot(token_keep_rate_eventful, gflops_stgt, c='green', label='STGT', marker='*', markersize=20, linewidth=3.5)
axs[1].plot(token_keep_rate_ours, gflops_ours, c='orange', label='Ours', marker='*', markersize=20, linewidth=3.5)
axs[1].set_xlabel('Patch Keep Rate', size=20)
axs[1].set_ylabel('GFLOPs', size=20)
axs[1].tick_params(axis='x', labelsize=18)
axs[1].tick_params(axis='y', labelsize=18)
# axs[1].set_title('PGD-20 (CIFAR-10)', size=20, y=1.3)
axs[1].grid(True, which='both', linestyle='--', linewidth=1.5)
# axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
#           ncol=2, prop ={'size':20})
axs[2].plot(token_keep_rate_eventful, latency_eventful, c='navy', label='Eventful', marker='*', markersize=20, linewidth=3.5)
axs[2].plot(token_keep_rate_eventful, latency_stgt, c='green', label='STGT', marker='*', markersize=20, linewidth=3.5)
axs[2].plot(token_keep_rate_ours, latency_ours, c='orange', label='Ours', marker='*', markersize=20, linewidth=3.5)
axs[2].set_xlabel('Patch Keep Rate', size=20)
axs[2].set_ylabel('Latency (ms)', size=20)
axs[2].tick_params(axis='x', labelsize=18)
axs[2].tick_params(axis='y', labelsize=18)
# axs[2].set_title('AutoPGD (CIFAR-10)', size=20, y=1.3)
axs[2].grid(True, which='both', linestyle='--', linewidth=1.5)
# axs[2].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
#           ncol=2, prop ={'size':20})
axs[3].plot(token_keep_rate_eventful, memory_eventful, c='navy', marker='*', markersize=20, linewidth=3.5, label='Eventful')
axs[3].plot(token_keep_rate_eventful, memory_stgt, c='green', marker='*', markersize=20, linewidth=3.5, label='STGT')
axs[3].plot(token_keep_rate_ours, memory_ours, c='orange', marker='*', markersize=20, linewidth=3.5, label='Ours')
axs[3].set_xlabel('Patch Keep Rate', size=20)
axs[3].set_ylabel('Memory (MB)', size=20)
axs[3].tick_params(axis='x', labelsize=18)
axs[3].tick_params(axis='y', labelsize=18)
# axs[3].set_title('CIFAR-100', size=20, y=1.3)
axs[3].grid(True, which='both', linestyle='--', linewidth=1.5)
# axs[3].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
#           ncol=2, prop ={'size':20})
# Manually define handles and labels for the legend
handles = [plt.Line2D([0], [0], color='navy', marker='*', markersize=20, linewidth=3.5, label='Eventful'),
           plt.Line2D([0], [0], color='green', marker='*', markersize=20, linewidth=3.5, label='STGT'),
           plt.Line2D([0], [0], color='orange', marker='*', markersize=20, linewidth=3.5, label='Ours')]

# Create a single legend
fig.legend(handles=handles, loc='upper center', ncol=3, prop ={'size':20})
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('plots/eventful_vs_ours.png')


# DETR Figure
token_keep_rate = [1.0, 0.57, 0.54, 0.38, 0.31, 0.29, 0.28]
map50 = [76.6, 77.9, 78.0, 77.7, 76.3, 76.5, 76.3]

# Sample data
x = np.arange(len(token_keep_rate))  # the label locations

width = 0.2  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
rects1 = ax.bar(x, token_keep_rate, label='FT-Full', color='olive', alpha=0.5)
ax.bar_label(rects1, padding=1, size=16)
# Adding labels and title
ax.set_ylabel('Patch keep rate', size=24)
ax.tick_params(axis='y', labelsize=18)
ax.set_xticks([])
ax.set_ylim([0.2, 1.1])

ax2 = ax.twinx()
ax2.plot(x, map50, c='navy', label='Eventful', marker='*', markersize=20, linewidth=3.5)
ax2.set_ylabel('mAP-50', size=20, color='navy')
ax2.tick_params(axis='y', labelsize=18, colors='navy')
ax2.set_ylim([70, 80])

plt.grid()
plt.tight_layout()
plt.savefig('plots/detr_map.png')

# DETR Latency
token_keep_rate = [1.0, 0.5, 0.4, 0.3, 0.2]
latency = [9.42, 8.45, 8.11, 7.795, 7.355]
x = np.arange(len(token_keep_rate))  # the label locations

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
rects1 = ax.bar(x, token_keep_rate, label='FT-Full', color='olive', alpha=0.5)
ax.bar_label(rects1, padding=1, size=16)
# Adding labels and title
ax.set_ylabel('Patch keep rate', size=24)
ax.tick_params(axis='y', labelsize=18)
ax.set_xticks([])
ax.set_ylim([0.1, 1.2])

ax2 = ax.twinx()
ax2.plot(x, latency, c='maroon', label='Eventful', marker='*', markersize=20, linewidth=3.5)
ax2.set_ylabel('Latency (ms)', size=20, color='maroon')
ax2.tick_params(axis='y', labelsize=18, colors='maroon')
ax2.set_ylim([6.8, 9.6])

plt.grid()
plt.tight_layout()
plt.savefig('plots/detr_latency.png')


# EViT Figure
token_rate_evit = [0.41, 0.47, 0.54, 0.63]
map_evit = [15.38, 33.16, 47.57, 59.24]
token_rate_ours = [0.66, 0.61, 0.47]
map_ours = [78.16, 77.77, 75.64]
token_rate_evit_ours = [0.51, 0.54, 0.59, 0.65]
map_evit_ours = [78.14, 78.21, 78.6, 78.95]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.scatter(token_rate_evit, map_evit, c='navy', label='EViT', marker='*', s=250)
ax.scatter(token_rate_ours, map_ours, c='orange', label='Ours', marker='^', s=200)
ax.scatter(token_rate_evit_ours, map_evit_ours, c='green', label='EViT+Ours', marker='s', s=200)

ax.set_xlabel('Patch Keep Rate', size=20)
ax.set_ylabel('mAP-50', size=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# ax.set_title('CIFAR-100', size=20, y=1.3)
ax.grid(True, which='both', linestyle='--', linewidth=1.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.8, 0.0),
          ncol=1, prop ={'size':16})

plt.tight_layout()
plt.savefig('plots/evit.png')


# Intro figure
map_evit = [15.38, 33.16, 47.57, 59.24]
latency_evit = [51.9, 59.84, 66.59, 79.5]
map_ours = [82.11, 81.9, 81.81, 81.78, 81.68, 81.06, 80.51]
latency_ours = [42.15, 40.89, 39.89, 38.95, 37.63, 37.03, 35.71]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# ax.scatter(latency_evit, map_evit, c='navy', label='EViT', marker='*', s=250)
ax.scatter(latency_eventful, map_eventful, c='green', label='Eventful', marker='s', s=200)
ax.scatter(latency_stgt, map_stgt, c='maroon', label='STGT', marker='o', s=200)
ax.scatter(latency_ours, map_ours, c='orange', label='Ours', marker='^', s=200)

ax.set_xlabel('Latency (ms)', size=20)
ax.set_ylabel('mAP-50', size=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# ax.set_title('CIFAR-100', size=20, y=1.3)
ax.grid(True, which='both', linestyle='--', linewidth=1.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.8, 0.0),
          ncol=1, prop ={'size':16})

plt.tight_layout()
plt.savefig('plots/intro.png')