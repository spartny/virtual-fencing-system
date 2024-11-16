import cv2 
import math

# initialize the list of reference point 
ref_points = [] 
crop = False
start = True
startPoint = []
points = []
edges = []
# from docx import Document
# doc = Document()
doc = open("outputs.txt", 'w')
preview = False
imageClosed = False


# method to draw the fence over the frame
def drawFence(image):
    file = open('fence.txt', 'r')
    fence = eval(file.read())
    file.close()
    for edge in fence:
        (x1, y1), (x2, y2) = edge
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# method to check whether a given point is inside the fence or not
def checkInside(xp, yp, track_id, fence, angle_vector, danger_vector=180.0, tolerance=5.0):
    """
    Determines whether a given point (xp, yp) is inside a defined fence (polygon) 
    and checks the direction of movement based on the object's angle vector.
    
    This function performs two main tasks:
    
    1. **Point-in-Polygon Check**: 
       It uses the ray-casting algorithm to determine whether a point lies inside a polygon.
       The polygon (fence) is stored in a file ('fence.txt') and is dynamically read during execution.

    2. **Direction Check**: 
       If the point is inside the fence, it calculates the object's angle vector using the past 
       coordinates of the object (identified by `track_id`). It then compares the angle vector to a 
       predefined `danger_vector` (default is 270 degrees) and checks whether the object is moving 
       within the dangerous angle range (± `tolerance` degrees).

    Args:
        xp (int): The x-coordinate of the point (usually the foot of the bounding box).
        yp (int): The y-coordinate of the point.
        track_id (int or tensor): Unique identifier for the tracked object.
        past_coordinates (dict): Dictionary storing past coordinates for each track_id.
        danger_vector (float, optional): The direction of movement considered dangerous (default is 270 degrees).
        tolerance (float, optional): A value that ± of the danger_vector is considered as a slice that is dangerous.
        
    Returns:
        bool: 
            - `True` if the point is inside the fence and the object is moving within the dangerous angle range.
            - `False` if the point is outside the fence or moving outside the dangerous angle range.

    Raises:
        ValueError: If there is an issue with the data or angle calculation.
    
    Notes:
        - The fence data is expected to be a list of edge coordinates and is loaded from 'fence.txt'.
        - Ensure that the past coordinates are being updated properly in the calling code to maintain accurate tracking.
    """
    
    cnt = 0
    result = None
    for edge in fence:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    if cnt % 2 == 1:
        #angle_vector = calculate_angle_vector(track_id, past_coordinates)
        # 175 <= angle <= 365 
        print(track_id, "Reached angle vector 175 <= angle <= 185", angle_vector)
        angle_message = f"{track_id} Reached angle vector 175 <= angle <= 185, Angle Vector: {int(angle_vector)}"
        doc.write("\n" + angle_message)
        if ((danger_vector - tolerance) <= int(angle_vector) ) and (int(angle_vector) <= (danger_vector + tolerance)):
            print("INSIDE")
            inside_message = "INSIDE"
            doc.write("\n" + inside_message)
            result = True
    else:
        # print('OUTSIDE')
        result = False
    doc.flush()
    return result

# method to build the virtual fence as per user input and save the fence
# context as a .text file 
            
def fenceBuild(image):
    def shape_selection(event, x, y, flags, param): 
        
        global ref_points, crop, start, startPoint, edges, points, preview, imageClosed
    
        if event == cv2.EVENT_LBUTTONDOWN:
            print("imageClose", imageClosed)
            if(not imageClosed):
                # preview = False
                ref_points.append((x, y))
                points.append([x, y])
                
                if start:
                    startPoint = [x,y]
                    start = False
                    
                else:
                    ref_points.append((x, y)) 

                    cv2.line(image, ref_points[0], ref_points[1], (0, 255, 255), 2)
                    # print(ref_points)
                    
                    prevPoint = ref_points[1]
                    ref_points = []
                    # print("POINTS",points)
                    ref_points.append(prevPoint)
                    # print(startPoint)
                    cv2.imshow("Virtual Fence Definition", image) 
                    # preview = True
                
                if len(points) >= 2:
                    edge = (points[-2], points[-1])
                    edges.append(edge)
                    # print("EDGES", edges)
                
                if points[-1][0] in range(startPoint[0] - 5, startPoint[0] + 5) and points[-1][1] in range(startPoint[1] -5, startPoint[1] + 5) and len(points) != 1:
                    print("Fence Built!")    
                    imageClosed = True

                    file = open('fence.txt', 'w')
                    file.write(str(edges))
                    file.flush()
                    file.close()
                    # cv2.putText(image, 'VIRTUAL FENCE DEFINED', (150, 300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
                    return
                
                

    cv2.namedWindow("Virtual Fence Definition") 
    cv2.setMouseCallback("Virtual Fence Definition", shape_selection) 
    cv2.imshow('Virtual Fence Definition', image)
    cv2.waitKey(0)


def update_coordinates(track_id, current_coords, past_coordinates):
    """
    Update the past coordinates for a given track ID.

    Parameters:
        track_id (int): The ID of the tracked object.
        current_coords (tuple): The (x, y) coordinates of the current position.
        past_coordinates (dict): A dictionary to keep track of past coordinates.

    Returns:
        None
    """

    # Convert the tensor track_id to an integer
    track_id_int = int(track_id.item()) if track_id is not None else 'None' 

    if track_id_int not in past_coordinates:
        # Initialize with an empty list for two previous coordinates
        past_coordinates[track_id_int] = []
    
    # Update past coordinates, keeping only the last two entries
    past_coordinates[track_id_int].append(current_coords)
    if len(past_coordinates[track_id_int]) > 75:
        past_coordinates[track_id_int].pop(0)  # Keep only the last two coordinates



def calculate_angle_vector(track_id, past_coordinates, reference='x'):
    """
    Calculate the angle vector based on the track ID using past coordinates.

    Parameters:
        track_id (int): The ID of the tracked object.
        past_coordinates (dict): A dictionary to keep track of past coordinates.
        reference (str): The reference direction for angle calculation ('x' for positive x-axis or 'y' for positive y-axis).

    Returns:
        float: The angle in degrees with respect to the specified reference direction.
    """

    # Convert the tensor track_id to an integer
    track_id_int = int(track_id.item()) if track_id is not None else 'None'

    # Check if the track ID has at least two sets of coordinates
    if track_id_int not in past_coordinates or len(past_coordinates[track_id_int]) < 2:
        return -1.0

    # Get the available coordinates (up to 5)
    coords = past_coordinates[track_id_int]  # Only take the last 10 coordinates if available
    
    # Initialize the total change in coordinates
    total_dx, total_dy = 0, 0

    # Calculate the sum of differences for all available coordinate pairs
    for i in range(1, len(coords)):
        previous_coords = coords[i-1]
        current_coords = coords[i]
        total_dx += current_coords[0] - previous_coords[0]
        total_dy += current_coords[1] - previous_coords[1]

    # Calculate the average change in coordinates
    avg_dx = total_dx / (len(coords) - 1)
    avg_dy = total_dy / (len(coords) - 1)

    if reference == 'x':
        # Calculate angle with respect to positive X-axis
        angle = math.atan2(avg_dy, avg_dx)  # Result in radians
    elif reference == 'y':
        # Calculate angle with respect to positive Y-axis
        angle = math.atan2(avg_dx, avg_dy)  # Swap dx and dy for Y-axis reference
    else:
        raise ValueError("Reference must be 'x' or 'y'")

    # Convert angle to degrees and normalize to [0, 360) degrees
    angle_degrees = math.degrees(angle) % 360

    return angle_degrees

 

if '__name__' == "__main__":
    print("main")
    
 
