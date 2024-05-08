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


# method to draw the fence over the frame
def drawFence(image):
    file = open('fence.txt', 'r')
    fence = eval(file.read())
    file.close()
    for edge in fence:
        (x1, y1), (x2, y2) = edge
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# method to check whether a given point is inside the fence or not
def checkInside(xp, yp):
    file = open('fence.txt', 'r')
    fence = eval(file.read())
    file.close()
    cnt = 0
    result = None
    for edge in fence:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1
    if cnt % 2 == 1:
        print("INSIDE")
        result = True
    else:
        print('OUTSIDE')
        result = False
    return result

# method to build the virtual fence as per user input and save the fence
# context as a .text file 
            
def fenceBuild(image):
    def shape_selection(event, x, y, flags, param): 
        
        global ref_points, crop, start, startPoint,imageClosed, edges, points, preview
    
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

                    cv2.line(image, ref_points[0], ref_points[1], (0, 0, 255), 2)
                    print(ref_points)
                    
                    prevPoint = ref_points[1]
                    ref_points = []
                    print("POINTS",points)
                    ref_points.append(prevPoint)
                    print(startPoint)
                    cv2.imshow("Virtual Fence Definition", image) 
                    # preview = True
                
                if len(points) >= 2:
                    edge = (points[-2], points[-1])
                    edges.append(edge)
                    print("EDGES", edges)
                
                if points[-1][0] in range(startPoint[0] - 5, startPoint[0] + 5) and points[-1][1] in range(startPoint[1] -5, startPoint[1] + 5) and len(points) != 1:
                    print("Image Closed")    
                    imageClosed = True

                    file = open('fence.txt', 'w')
                    file.write(str(edges))
                    file.flush()
                    file.close()
                    cv2.putText(image, 'VIRTUAL FENCE DEFINED', (150, 300), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
                    return
                
                

    cv2.namedWindow("Virtual Fence Definition") 
    cv2.setMouseCallback("Virtual Fence Definition", shape_selection) 
    cv2.imshow('Virtual Fence Definition', image)
    cv2.waitKey(0)
# image = cv2.imread('Images/cat_dog.jpg')
# image = cv2.imread('Images\\parking-lot.jpg')
# image = cv2.imread('Images\\exported frames\\day_frame_880.jpg')

if '__name__' == "__main__":
    print("main")
    
 
