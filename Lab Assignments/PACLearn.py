import matplotlib.pyplot as plt
import numpy as np

# Define the feature space
X = np.array([[1, 1], [2, 2], [3, 1], [4, 2]])

# Define labels for each point
labels = [1, -1, 1, -1]  # Positive: 1, Negative: -1

# Define function to plot points and rectangles
def plot_points_and_rectangles(rectangles):
    plt.figure(figsize=(8, 6))
    # Plot points
    for i, label in enumerate(labels):
        color = 'blue' if label == 1 else 'red'
        plt.scatter(X[i, 0], X[i, 1], color=color, s=100)
    # Plot rectangles
    for rect in rectangles:
        plt.gca().add_patch(rect)
    plt.xlim(0, 5)
    plt.ylim(0, 3)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('C1 Shattering Points')
    plt.grid(True)
    plt.show()

# Define function to generate axis-parallel rectangles
def generate_rectangles():
    rectangles = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            x_min = min(X[i][0], X[j][0])
            x_max = max(X[i][0], X[j][0])
            y_min = min(X[i][1], X[j][1])
            y_max = max(X[i][1], X[j][1])
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='green', facecolor='none')
            rectangles.append(rect)
    return rectangles

# Plot the points and rectangles
plot_points_and_rectangles(generate_rectangles())

import itertools

# Define function to check if C2 can shatter points
def can_shatter_c2():
    for subset in itertools.combinations(X, 2):
        # Check if there exists a subset that separates the points
        x1, x2 = subset
        if (x1[0] < x2[0] and x1[1] < x2[1]) or (x1[0] > x2[0] and x1[1] > x2[1]):
            return False
    return True

# Check if C2 can shatter points
can_shatter = can_shatter_c2()
print("C2 can shatter points:", can_shatter)
