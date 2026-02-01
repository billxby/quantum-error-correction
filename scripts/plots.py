import matplotlib.pyplot as plt
import numpy as np

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 44
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['font.weight'] = 'bold'

# Data - ordered consistently: GCN, GAT, GIN, GraphSAGE
models = ['GCN', 'GAT', 'GIN', 'GraphSAGE']
accuracy = [73.4, 81.6, 83.3, 86.9]
inference_speed = [8901, 8458, 13270, 13287]  # throughput (graphs/s)

# Color palette matching the image
# GCN = Blue, GAT = Purple/Pink, GIN = Orange/Peach, GraphSAGE = Green
model_colors_accuracy = ['#ADD8E6', '#EBC6DF', '#FED8B1', '#D4EBD0']
model_colors_speed = ['#ADD8E6', '#EBC6DF', '#FED8B1', '#D4EBD0']


# Create figure and axis
fig, ax1 = plt.subplots(figsize=(20, 12))


# Set the width of bars and positions
x = np.arange(len(models))
width = 0.35
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=32, fontweight='bold')

# Create bars for accuracy (left y-axis) with model-specific colors
bars1 = []
for i, (pos, acc, color) in enumerate(zip(x, accuracy, model_colors_accuracy)):
    bar = ax1.bar(pos - width/2, acc, width,
                  color=color, edgecolor='black', linewidth=3, alpha=0.9)
    bars1.append(bar)

# Configure left y-axis for accuracy
ax1.set_xlabel('GNN Model', fontsize=40, fontweight='bold', labelpad=20)
ax1.set_ylabel('Accuracy (%)', fontsize=40, fontweight='bold', labelpad=20, color='black')
ax1.set_ylim([0, 100])


# Add accuracy values on top of bars
for i, (bar_container, val) in enumerate(zip(bars1, accuracy)):
    bar = bar_container[0]
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=30, fontweight='bold', color='black')

# Create second y-axis for inference speed
ax2 = ax1.twinx()

# Create bars for inference speed (right y-axis) with model-specific colors
bars2 = []
for i, (pos, speed, color) in enumerate(zip(x, inference_speed, model_colors_speed)):
    bar = ax2.bar(pos + width/2, speed, width,
                  color=color, edgecolor='black', linewidth=3, alpha=0.9)
    bars2.append(bar)

# Configure right y-axis for inference speed
ax2.set_ylabel('Inference Speed (graphs/s)', fontsize=40, fontweight='bold',
               labelpad=20, color='black')
ax2.set_ylim([0, max(inference_speed) * 1.2])

# Add inference speed values on top of bars
for i, (bar_container, val) in enumerate(zip(bars2, inference_speed)):
    bar = bar_container[0]
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 400,
            f'{val:,}',
            ha='center', va='bottom', fontsize=30, fontweight='bold', color='black')

# Add title
ax1.set_title('Model Performance: Accuracy and Inference Speed',
              fontsize=48, fontweight='bold', pad=30)

# Create custom legend with colored patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=model_colors_accuracy[0], edgecolor='black', linewidth=2, label='Accuracy'),
    Patch(facecolor=model_colors_speed[0], edgecolor='black', linewidth=2, label='Inference Speed')
]
ax1.legend(handles=legend_elements,
          loc='upper left', fontsize=32, frameon=True,
          fancybox=True, shadow=True, framealpha=0.95)

# Add grid for better readability
ax1.set_axisbelow(True)

ax1.set_yticks([])
ax2.set_yticks([])


# Adjust layout
# plt.tight_layout()

# Save figure
plt.savefig('../',
            dpi=300, bbox_inches='tight', facecolor='white')

print("Combined performance plot saved successfully!")
plt.show()