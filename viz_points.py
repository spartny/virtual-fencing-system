import matplotlib.pyplot as plt



# Define the coordinates with center as origin (0, 0) and foot_x, foot_y 10 away from origin

foot_x, foot_y = 0, -10

center_x, center_y = 0, 0



current_points = [
    (foot_x - 6, center_y),        
    (center_x + 2, center_y - 1),
    (center_x - 3, center_y + 5),
    (center_x - 6, center_y - 4),
    (center_x, center_y),  
    (center_x + 7, center_y),
    (foot_x, foot_y),
    (center_x, center_y + 5),
    (center_x + 6, center_y - 2),
    (center_x - 4, center_y + 7),
    (center_x - 5, center_y + 5),
    (center_x + 8, center_y - 3),
    (center_x - 2, center_y + 5),
    (center_x - 8, center_y + 9),
    (center_x + 4, foot_y - 2)
]



# Extract x and y coordinates

x_coords, y_coords = zip(*current_points)



# Plot the points

plt.figure(figsize=(8, 8))

plt.scatter(x_coords, y_coords, color='blue', label='Points')

plt.scatter([center_x], [center_y], color='red', label='Center (0, 0)')

plt.scatter([foot_x], [foot_y], color='green', label='Foot Point')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

plt.gca().set_aspect('equal', adjustable='box')



# Annotate points

for i, (x, y) in enumerate(current_points):

    plt.text(x, y, f"{i}", fontsize=8, ha='right')



plt.title("Visualization of Points with Center as Origin")

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.legend()

plt.grid()

plt.show()