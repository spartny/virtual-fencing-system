import cv2 
  
# now let's initialize the list of reference point 
ref_points = [] 
crop = False
start = True
startPoint = []
points = []
edges = []

preview = False
imageClosed = False

def checkInside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    if cnt%2 == 1:
        print("INSIDE")
    else:
        print('OUTSIDE')

def shape_selection(event, x, y, flags, param): 
    global ref_points, crop, start, startPoint, imageClosed, edges, points, preview
    
    
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
            checkInside(edges, x, y)
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if preview:
    #         previewImg = image.copy()
    #         if len(points) >= 2:
    #             cv2.line(previewImg, points[-2], points[-1], [0, 0 ,255], 2)
  
  

image = cv2.imread('Virtual-Fencing-Home-Security-System\Images\cat_dog.jpg')

cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection) 
  
  
cv2.imshow('image', image)
cv2.waitKey(0)
# close all open windows 
