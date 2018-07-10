from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import math
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
##
def center(img):
    top=-1
    bot=-1
    left=-1
    right=-1
    i=int(img.shape[1]/2)
    while top==-1 and i>=0:
        if sum(img[i,0:img.shape[1]])/255<=0:
               top=0
        else:
               i=i-1
    j=int(img.shape[1]/2)
    while bot==-1 and j<img.shape[1]:
        if sum(img[j,0:img.shape[1]])/255<=0:
               bot=0
        else:
               j=j+1
    k=int(img.shape[0]/2)
    while right==-1 and k<img.shape[0]:
        if sum(img[0:img.shape[0],k])/255<=0:
               right=0
        else:
               k=k+1
    m=int(img.shape[0]/2)
    while left==-1 and m<img.shape[0]:
        if sum(img[0:img.shape[1],m])/255<=0:
               left=0
        else:
               m=m-1
    newcell=np.zeros((img.shape[1],img.shape[0]),np.uint8)
##    print(i,j,k,m)
    
    newcell[int(img.shape[1]/2)-math.floor((j-i)/2):int(img.shape[1]/2)+math.ceil((j-i)/2),int(img.shape[0]/2)-math.floor((k-m)/2):int(img.shape[0]/2)+math.ceil((k-m)/2)]=img[i:j,m:k]
    return newcell
###########################################################
def find_empty_location(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                l[0]=row
                l[1]=col
                return True
    return False
 
# Returns a boolean which indicates whether any assigned entry
# in the specified row matches the given number.
def used_in_row(arr,row,num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False
 
# Returns a boolean which indicates whether any assigned entry
# in the specified column matches the given number.
def used_in_col(arr,col,num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False
 
# Returns a boolean which indicates whether any assigned entry
# within the specified 3x3 box matches the given number
def used_in_box(arr,row,col,num):
    for i in range(3):
        for j in range(3):
            if(arr[i+row][j+col] == num):
                return True
    return False
 
# Checks whether it will be legal to assign num to the given row,col
#  Returns a boolean which indicates whether it will be legal to assign
#  num to the given row,col location.
def check_location_is_safe(arr,row,col,num):
     
    # Check if 'num' is not already placed in current row,
    # current column and current 3x3 box
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num)
 
# Takes a partially filled-in grid and attempts to assign values to
# all unassigned locations in such a way to meet the requirements
# for Sudoku solution (non-duplication across rows, columns, and boxes)
def solve_sudoku(arr):
     
    # 'l' is a list variable that keeps the record of row and col in find_empty_location Function    
    l=[0,0]
     
    # If there is no unassigned location, we are done    
    if(not find_empty_location(arr,l)):
        return True
     
    # Assigning list values to row and col that we got from the above Function 
    row=l[0]
    col=l[1]
     
    # consider digits 1 to 9
    for num in range(1,10):
         
        # if looks promising
        if(check_location_is_safe(arr,row,col,num)):
             
            # make tentative assignment
            arr[row][col]=num
 
            # return, if sucess, ya!
            if(solve_sudoku(arr)):
                return True
 
            # failure, unmake & try again
            arr[row][col] = 0
             
    # this triggers backtracking
    return False

####################################################        
img1 = cv2.imread('sudoku.jpg',0)
img=cv2.GaussianBlur(img1,(11,11), 0);
ret,thres=cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)
athres = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
##cv2.imshow('athres',athres)
##cv2.waitKey(0)
kernel = np.array([(0,1,0),(1,1,1),(0,1,0)],np.uint8)
dilation = cv2.dilate(athres,kernel,iterations = 1)
##final=cv2.erode(dilation,kernel,iterations=1)
max1=-1
pt=[0,0]
[height,width]=dilation.shape

mask=np.zeros((height+2,width+2),dtype=np.uint8)
for i in range(0,width):
    for j in range(0,height):
        if dilation[j,i]==255:
            x=cv2.floodFill(dilation,mask,(i,j),64)
            if x[0]>max1:
                max1=x[0]
                pt=[i,j]
mask1=np.zeros((height+2,width+2),dtype=np.uint8)#really dont know why this was required but had to be done,without it the image after floodFill was same as before
cv2.floodFill(dilation,mask1,(pt[0],pt[1]),255)
##cv2.imshow('flood',dilation)
##cv2.waitKey(0)
for i in range(0,width):
    for j in range(0,height):
        if(dilation[j,i]==64):
            cv2.floodFill(dilation,mask1,(i,j),0)
erosion=cv2.erode(dilation,kernel,iterations=1)
lines = cv2.HoughLines(erosion,1,np.pi/180, 200)
##print(lines)
for vec1 in lines:
    vec1=vec1[0]
    if vec1[0]==0 and vec1[1]==-100:
        continue
    else:
        for vec2 in lines:
            vec2=vec2[0]
            if (vec2[0]==vec1[0] and vec2[1]==vec1[1]) or (vec2[0]==0 and vec2[1]==-100):
                continue
            else:
                if abs(vec1[0]-vec2[0])<20 and abs(vec1[1]-vec2[1])<np.pi*10/180:
                    vec1[0]=(vec1[0]+vec2[0])/2
                    vec1[1]=(vec1[1]+vec2[1])/2
                    vec2[0]=0
                    vec2[1]=-100
                else:
                    continue
collector=[-1,-1,-1,-1,-1]
a=max(erosion.shape[0],erosion.shape[1])
b=max(erosion.shape[0],erosion.shape[1])
c=max(erosion.shape[0],erosion.shape[1])
d=max(erosion.shape[0],erosion.shape[1])
for i in range(len(lines)):
   
    vec=lines[i]
    vec=vec[0]
    if vec[0]==0 and vec[1]==-100:
        continue
    if vec[1]!=0:
        a1=vec[0]/np.sin(vec[1])
        c1=erosion.shape[0]-abs(vec[0]/np.sin(vec[1]))
    else:
        a1=max(erosion.shape[0],erosion.shape[1])+100
        c1=max(erosion.shape[0],erosion.shape[1])+100
    if vec[1]!=np.pi/2 or vec[1]!=-np.pi/2:
        b1=vec[0]/np.cos(vec[1])
        d1=erosion.shape[1]-abs(vec[0]/np.cos(vec[1]))
    else:
        b1=max(erosion.shape[0],erosion.shape[1])+100
        c1=max(erosion.shape[0],erosion.shape[1])+100
    if abs(a1)<a and a1>0:
        a=a1
        collector[0]=i
    if abs(b1)<b and b1>0:
        b=b1
        collector[1]=i
    if abs(c1)<c and c1>0:
        c=c1
        collector[2]=i
    if abs(d1)<d and d1>0:
        d=d1
        collector[3]=i
for i in range(len(lines)):
    xnew=0
    vec=lines[i]
    vec=vec[0]
    for j in range(4):
        if collector[j]==i:
            xnew=1
            break
    if xnew!=1:
        vec[0]=0
        vec[1]=-100
points=[0,0,0,0,0]
collector[4]=collector[0]
##print(lines)
for i in range(4):
    
    vec1=lines[collector[i]]
    vec1=vec1[0]
    vec2=lines[collector[i+1]]
    vec2=vec2[0]
    if vec1[1]==0 and vec2[1]!=0:
        x=vec1[0]*np.cos(vec1[1])
        y=vec2[0]/np.sin(vec2[1])-x*np.cos(vec2[1])/np.sin(vec2[1])
    elif vec1[1]!=0 and vec2[1]==0:
        x=vec2[0]*np.cos(vec2[1])
        y=vec1[0]/np.sin(vec1[1])-x*np.cos(vec1[1])/np.sin(vec1[1])
    elif vec1[1]!=0 and vec2[1]!=0:
        x=(vec2[0]/np.sin(vec2[1])-vec1[0]/np.sin(vec1[1]))/(np.cos(vec2[1])/np.sin(vec2[1])-np.cos(vec1[1])/np.sin(vec1[1]))
        y=vec1[0]/np.sin(vec1[1])-x*np.cos(vec1[1])/np.sin(vec1[1])
    points[i]=[int(x),int(y)]
points[4]=points[0]
##print(points)
maxlen=0
for i in range(4):
    if maxlen<((points[i][0]-points[i+1][0])**2+(points[i][1]-points[i+1][1])**2):
        maxlen=(points[i][0]-points[i+1][0])**2+(points[i][1]-points[i+1][1])**2 
pts1=np.float32(points[0:4])
maxlen=int(maxlen**0.5)
pts2=np.float32([[0,0],[0,maxlen],[maxlen,maxlen],[maxlen,0]])
###################################################################
M=cv2.getPerspectiveTransform(pts1,pts2)
cropped=np.mat((img1.shape[1],img1.shape[0]),np.float32)
cropped=cv2.warpPerspective(img1,M,(maxlen,maxlen))
##cv2.imshow('inter',cropped)
##cropped=cv2.erode(cropped,kernel,iterations=1)
##ret,cropped=cv2.threshold(cropped,120,255,cv2.THRESH_BINARY)
cropped=cv2.adaptiveThreshold(cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,1)
##kernel2=np.mat([1],np.float32)
##cropped=cv2.erode(cropped,kernel,iterations=1)
cv2.imshow('intermediate',cropped)
##cv2.imshow('cropped',cropped)
##cv2.waitKey(0)
from mnist import MNIST
import cv2
import numpy as np
mndata = MNIST('data')
images, labels = mndata.load_training()
images=np.array(images)
labels=np.array(labels)
#########################################################
hog_list=[]
for image in images:
    fd = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    hog_list.append(fd)
hog_list= np.array(hog_list, 'float64')
clf = LinearSVC()
clf.fit(hog_list,labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)
############################################################
##cv2.imshow('img',cropped)
##cv2.waitKey(8000)
##print(images[0])
##for i in range(len(images)):
##    images[i]=np.mat(images[i],np.uint8)
##    images[i]=images[i].reshape((28,28))
####    ret,images[i] = cv2.threshold(images[i],20,255,cv2.THRESH_BINARY)
####    images[i]=cv2.resize(images[i], (200,200))
##    cv2.imshow('img1',images[i])
##    cv2.waitKey(50)
##print(images[0])
dist=int(maxlen/9)
curcell=np.ones((dist,dist),np.uint8)
curcell=cropped[0:dist,0:dist]
##import train
##knn = KNeighborsClassifier(n_neighbors=1)
##knn.fit(images[0:5000],labels[0:5000])
mysudoku=[[],[],[],[],[],[],[],[],[]]
for j in range(9):
    for i in range(9):
        x=0
        y=0
##        print(i,j)
        curcell=cropped[j*dist:j*dist+dist,i*dist:i*dist+dist]
        curcell=cv2.resize(curcell,(28,28),interpolation=cv2.INTER_AREA)
##        ret,curcell = cv2.threshold(curcell,120,255,cv2.THRESH_BINARY)
##        mask2=np.zeros((curcell.shape[0]+2,curcell.shape[1]+2),dtype=np.uint8)
##        for p in range(curcell.shape[0]):
##            if curcell[p,0]==255:
##                cv2.floodFill(curcell,mask2,(p,0),0)
##            if curcell[p,curcell.shape[1]-1]==255:
##                cv2.floodFill(curcell,mask2,(curcell.shape[1]-1,p),0)
##        for q in range(curcell.shape[1]):
##            if curcell[0,q]==255:
##                cv2.floodFill(curcell,mask2,(q,0),0)
##            if curcell[curcell.shape[0]-1,q]==255:
##                cv2.floodFill(curcell,mask2,(q,curcell.shape[0]-1),0)
        cell=np.zeros((28,28),np.uint8)
        cell[4:26,4:26]=curcell[4:26,4:26]
        cell=center(cell)
##        cell=cv2.erode(cell,kernel,iterations=1)
        cv2.imshow('curcell',cell)
        cv2.waitKey(500)
##        cell=np.resize(cell,(784,))

####        print(np.sum(curcell==255))
        if np.sum(cell>=255)<(5):
            mysudoku[i].append(0)
            print(0)
            continue
        else:
            hog_fd = hog(cell, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            mysudoku[i].append(clf.predict(np.array([hog_fd], 'float64'))[0])
            print(clf.predict(np.array([hog_fd], 'float64'))[0])
##########################################################
solve_sudoku(mysudoku)
print(mysudoku)
###########################################################


            
##            cell=cell.reshape(1,-1)
##            x=knn.predict_proba(cell)[0]
####            print(x)
##            x=x.tolist()
##            print(x.index(max(x)))
##        a=np.zeros((59999,))
##        for k in range(59999):
##            if(labels[k]==0):
##                a[k]=100000
##                continue
##            a[k]=np.sum(np.square((images[k]-tatti)))
####            a[k]=[a[k],labels[k]]
####        print(len(a))
####        print(labels[np.argmin(a)])            
##        z=np.argsort(a)
##        z=z[0:10]
####        print(z)
##        y=np.zeros((10,))
##        for p in range(10):
##            y[p]=labels[z[p]]
####        print(y)
##        print(y[np.argmax(y)])
##        a=[]
##        for k in range(5000):
##            if labels[k]==0:
##                continue
##            if np.sum(np.square((images[k]-tatti)/255))<65:
##                a.append(labels[k])
##        print(len(a))
##        print(np.bincount(a).argmax())
# train using K-NN


##print(lines)               ##DRAW LINES
##for vec in lines:
##    vec=vec[0]
##    if vec[0]!=0 or vec[1]!=-100:
##        
##        if vec[1]==0 or vec[1]==180:
##            x=vec[0]*np.cos(vec[1])
##            cv2.line(erosion,(int(x),erosion.shape[0]),(int(x),0),(0,0,255),1)
##        else:
##            c=vec[0]/np.sin(vec[1])
##            m=-(np.cos(vec[1]))/(np.sin(vec[1]))
##
##            cv2.line(erosion,(0,int(c)),(erosion.shape[1],int((erosion.shape[1])*m+c)),(255,255,255),1)
##    else:
##        continue        
##                     if vec1[1]==0:
##                        x1=vec1[0]*np.cos(vec1[1])
##                        y1=erosion.shape[0]
##                        x2=vec1[0]*np.cos(vec1[1])
##                        y2=0
##                     else:
##                         
##                         x1=0
##                         y2=vec1[0]/np.sin(vec1[1])
##                         x2=erosion.shape[1]
##                         y2=(erosion.shape[1])*m+c
##                     if vec2[1]==0:
##                         x3=vec2[0]*np.cos(vec2[1])
##                         y3=erosion.shape[0]
##                         x4=vec2[0]*np.cos(vec2[1])
##                         y4=0
##                     else:
##                         x3=0
##                         y3=vec2[0]/np.sin(vec2[1])
##                         x4=erosion.shape[1]
##                         y4=(erosion.shape[1])*m+c
##                     x1fin=int((x1+x3)/2)
##                     y1fin=int((y1+y3)/2)
##                     x2fin=int((x2+x4)/2)
##                     y2fin=int((y1+y3)/2)
    














cv2.imshow('with_lines',erosion)
##cv2.imshow('sudoku final',final)
cv2.imshow('sudoku cropped',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
