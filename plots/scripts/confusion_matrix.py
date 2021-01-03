import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

labels = ["wake", "N1", "N2",  "N3", "REM"]


M = np.array([[613, 22, 37, 4, 22],
     [112, 69, 102, 1, 59],
     [193, 67, 2051, 78, 155],
     [139, 16, 295, 947, 20],
     [137, 62, 178, 0, 793]])

Mperc = np.zeros([5,5])

for i in range(len(labels)):
    for j in range(len(labels)):
        Mperc[i, j] = M[i, j] / np.sum(M[:, j])



plt.set_cmap('inferno')
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(Mperc)

# We want to show all ticks...
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set_xlabel("True label", size=18)
ax.set_ylabel("Prediction", size=18)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

def make_label(m, i, j):
    perc = m[i, j] / np.sum(m[:, j]) * 100.
    return "%.1f%% (%d)" % (perc, m[i, j])

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):

        text = ax.text(j, i, make_label(M, i, j),
                       ha="center", va="center", color="w")

        if (j == 3) & (i == 3):
            text = ax.text(j, i, make_label(M, i, j),
                           ha="center", va="center", color="black")

ax.set_title("Confusion matrix", fontweight="bold", size=22)
fig.tight_layout()
plt.savefig("confusion_matrix.pdf")
plt.show()
