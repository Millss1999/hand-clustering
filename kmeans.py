import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
from sklearn.svm import SVC
from skimage import io


def main():
    #hand
    hand = cv2.imread('image1.jpeg')
    h = hand.shape[0] #get height/width, 320px
    w = hand.shape[1] #240px
    count = 0 #counter
	
    #get k
    desiredK = input("Enter in desired k value: ")
	
    #final map
    #finalMap = np.zeros((h,w,100)) 
    map = np.zeros((h,w,1))
	
    #kmeans will iterate 200 times, each run returns mapping of pixels to clusters
    for i in range(100):
      pxAssign = kmeans(int(desiredK)) #get px assignments for iteration of kmeans (2D matrix)
      #pxAssign = pxAssign.reshape((pxAssign.shape[0], pxAssign.shape[1], 1)) #convert to 3D array
      
      #set array of px assignments to h,w,1
      map= np.dstack((map,pxAssign)) #append each 2D pxAssign to full map
      count+=1
    #np.set_printoptions(threshold=np.inf)
    
    #pdb.set_trace()
	#calculate frequency of assignments
    iCoor = ""
    jCoor = ""
    while jCoor != "EXIT":
      iCoor = input("Enter in i coordinate: ")
      jCoor = input("Enter in j coordinate ('EXIT' to exit): ")
      if jCoor == "EXIT":
        break
      else:
        print(pxFreq(iCoor,jCoor,map))
############################    
    #SVM (GR)
    #svmHand = io.imread('ideal-artificial-light')
    #manually sample image regions
    rows, cols, bands = hand.shape
    classes = {'bg': 0, 'hand': 1}
    n_classes = len(classes)
    palette = np.uint8([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) #define palette
    #begin supervised
    sv = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
	
	#select regions (height and width)
	#get background sample
    print("Enter in background sample regions")
    bgSamp = sample()
    print("Enter in hand sample regions")
    handSamp = sample()
    #set regions
    sv[bgSamp[0]:bgSamp[1], bgSamp[2]:bgSamp[3]] = classes['bg']
    sv[handSamp[0]:handSamp[1], handSamp[2]:handSamp[3]] = classes['hand']
	#fit 
    X = hand.reshape(rows*cols, bands)
    y = sv.ravel()
    train = np.flatnonzero(sv < n_classes)
    test = np.flatnonzero(sv == n_classes)
	#train/display
    clf = SVC(gamma='auto')
    clf.fit(X[train], y[train])
    y[test] = clf.predict(X[test])
    sv = y.reshape(rows, cols)

    io.imshow(palette[sv])
    plt.show()

#can return, array of which cluster each px is in, center values (RGB), segmented image
def kmeans(k):

   #read in image
   image = cv2.imread('image1.jpeg')

   #Convert BGR to RGB
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   plt.imshow(image)

   # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
   pixel_vals = image.reshape((-1,3)) 
  
   # Convert to float type 
   pixel_vals = np.float32(pixel_vals)
   #criteria
   criteria = (cv2.TERM_CRITERIA_MAX_ITER, 50,.95) # iteration

   #set k
   #k is passed in
   #kmeans
   retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  

   #covert data
   centers = np.uint8(centers) 
   segmented_data = centers[labels.flatten()]

   #reshape/show data
   # reshape data into the original image dimensions 
   segmented_image = segmented_data.reshape((image.shape)) #segmented_image = array of pixels-each val is rgb array
   plt.imshow(segmented_image)

   #plt.show()

   #print(centers)
   
   #print(labels) #labels=which cluster does pixel belong to (array of all px)
   #2 clusters- black[0], skin[1]
   if centers[0][0] > centers[1][0]: #if first cluster get assigned as skin
    for i in labels: 
     if i == 0: #if skin is labeled as 0
       i = 1 #swap 
     elif i == 1:
       i = 0	 
   #form labels into 2d array
   labels = np.reshape(labels, (-1, image.shape[1])) #h rows, w cols
   #print(segmented_image)
   return labels

#return frequency of assignment of specified pixel (i,j)
def pxFreq(i,j,finalMap):
  c0Count = 0;
  #iterate through 100 "sheets" of data at (i,j)
  for sheet in range(100):
    #if px was assigned to cluster 0 (black)
    if finalMap[int(i), int(j), int(sheet)] == 0:
      c0Count+=1
  #return freq. assignments of pixel
  result ="Px (" + str(i) + "," + str(j) + "):\n" + "Dark cluster freq: " + str(c0Count / 100) + "\nSkin cluster freq: " + str(1 - (c0Count/100)) 
  return result
  
#returns sampled image region
def sample():
  
  #sample region
  hStart = input("Input hStart: ")
  hEnd = input("Input hEnd: ")
  wStart = input("Input wStart: ")
  wEnd = input("Input wEnd: ")
  #croppedImg = image.crop((x1,y1,x2,y2)) #(x1,y1) is top left px of sample, (x2,y2) is bottom right px of sample
  return (int(hStart),int(hEnd),int(wStart),int(wEnd))
  
if __name__ =="__main__":
	main()