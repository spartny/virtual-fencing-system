import cv2 
  
# initialize the list of reference point 
ref_points = [] 
crop = False
start = True
startPoint = []
points = []
edges = []

preview = False
imageClosed = False

# read virtual fence context to check whether center of object is in the center or not
file = open('fence.txt', 'r')
fence = eval(file.read())
file.close()

def checkInside(image, xp, yp):
    file = open('fence.txt', 'r')
    fence = eval(file.read())
    file.close()
    cnt = 0
    result = None
    for edge in fence:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    if cnt%2 == 1:
        image = cv2.putText(image, 'BREACH', (150, 300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("image", image)
        print("INSIDE")
    else:
        image = cv2.putText(image, 'NO BREACH', (150, 300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("image", image)
        print('OUTSIDE')

# method to build the virtual fence as per user input and save the fence
# context as a .text file 
def shape_selection(event, x, y, flags, param): 
    global ref_points, crop, start, startPoint, imageClosed, edges, points, preview, fence
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if(not imageClosed):
            # preview = False
            ref_points.append((x, y))
            points.append([x, y])
            
            if start:
                startPoint = [x,y]
                start = False
                
            else:
                ref_points.append((x, y)) 

                cv2.line(image, ref_points[0], ref_points[1], (0, 0, 255), 2)
                print(ref_points)
                
                prevPoint = ref_points[1]
                ref_points = []
                print("POINTS",points)
                ref_points.append(prevPoint)
                print(startPoint)
                cv2.imshow("image", image) 
                # preview = True
            
            if len(points) >= 2:
                edge = (points[-2], points[-1])
                edges.append(edge)
                print("EDGES", edges)
            
            if points[-1][0] in range(startPoint[0] - 5, startPoint[0] + 5) and points[-1][1] in range(startPoint[1] -5, startPoint[1] + 5) and len(points) != 1:
                print("Image Closed")    
                imageClosed = True
        else:
            file = open('fence.txt', 'w')
            file.write(str(edges))
            file.flush()
            file.close()
            checkInside(image, x, y)
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if preview:
    #         previewImg = image.copy()
    #         if len(points) >= 2:
    #             cv2.line(previewImg, points[-2], points[-1], [0, 0 ,255], 2)
  
  

# image = cv2.imread('Images/cat_dog.jpg')
# image = cv2.imread('Images\\parking-lot.jpg')
image = cv2.imread('Images\\exported frames\\day_frame_880.jpg')

cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection) 
  
  
cv2.imshow('image', image)
cv2.waitKey(0)
# close all open windows 
