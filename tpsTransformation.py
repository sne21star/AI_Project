import csv
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.color import rgb2gray
'''
Inspiration to write the TPS class:
Followed formulas and variables from this paper: researchgate.net/publication/2364266_Approximate_Thin_Plate_Spline_Mappings
https://github.com/daeyun/TPS-Deformation
https://github.com/vladimir-ch/thin-plate-splines
https://github.com/cheind/py-thin-plate-spline
https://github.com/iwyoo/tf_ThinPlateSpline
https://github.com/WarBean/tps_stn_pytorch
'''
contoursList = []
'''
This function gets the contours from the text detection output. There are not inputs but you can change it to get the right txt file.
'''
def getContours():
  with open('/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/CUTE80/output/image001.txt', 'r') as fd:
    reader = csv.reader(fd)
    listElement = []
    for row in reader:
      listElement.append(row)
    
    index = 0
    temp = []
    contours = []
    for l in listElement[0]:
      if (index):
        x = int(l)
        temp.append(x)
        index = 0
        contours.append(temp)
        temp = []
      else:
        x = int(l)
        temp.append(x)
        index += 1
    return contours

'''
This function gets the files for the Text center line.
'''
def getFiles(pathToTCL):
  with open(pathToTCL, 'r') as fd:
      reader = csv.reader(fd)
      listElement = []
      for row in reader:
        temp = []
        #print(row)
        temp.append(int(row[0]))
        temp.append(int(row[1]))
        listElement.append(temp)
        #print(temp)
  return listElement

'''
This class represents the TPS transformation.
'''
class thinPSTransformation:
    '''
    L2NORM: Calculates the Distance between two matricies.
    '''
    @staticmethod
    def L2NORM(xyi, xy):
        topLeftMatrix = xyi[:, None, :2]
        bottomLeftMatrix = xy[None, :, :2]
        squareV = np.square(topLeftMatrix - bottomLeftMatrix ).sum(-1)
        sqrtV = np.sqrt(squareV)
        return sqrtV
    '''
    Bending energy equation. This equation represents the amount of difficulty it takes to map the coordinates.
    '''
    @staticmethod
    def U(rij):
        equationResult = pow(rij,2) * np.log(rij+1e-6)
        return equationResult

    '''
    This function calculates the theta needed to get the target points and source points transformation. This equation is used to solve the inverse of L. However, in my case the inverse of L could not always be solved due to the fact that sometimes the matrix was singular. Therefore I used least squares regression.
    '''
    @staticmethod
    def calcTheta(point_Coord):
        width = point_Coord.shape[0]
        K = thinPSTransformation.U(thinPSTransformation.L2NORM(point_Coord, point_Coord)) #K value in the TPS coefficients. Call the energy equation or U.
        P = np.ones((width, 3), dtype=np.float32) #the coordinates TPS coefficient
        P[:, 1:] = point_Coord[:, :2] #get the point coordinates and place in P so that now it is (1, x, y)
        P_T = P.T #get transpose of matrix.
        O = np.zeros((width+3, width+3), dtype=np.float32) #Zero matrix ).
        o = np.zeros(width+3, dtype=np.float32) #the width + 3 column of vector zeros
        o[:width] = point_Coord[:, -1] #input the change in x and y coordinates into vector with the rest of the values as zero
        O[:width, :width] = K #Take the O matrix and add the K, P, and P_T matrix in O matrix. K will be the top left of the final matrix.
        O[:width, -3:] = P #P will be the right matrix
        O[-3:, :width] = P_T #P transform will be the bottom left matrix
        t_Final = np.linalg.lstsq(O,o)[0] # Least sqaures regression to solve for L^-1 without calculating the acutal inverse
        return t_Final #return final theta
    '''
    This function calculates interpolant equation
    '''
    @staticmethod
    def interpoltantEquation(x, c, t):
        x = np.atleast_2d(x) #This ensures that the list is 2d list of x and y coordinates and will ensure that list is 2d even if it was not beffore.
        eucledianDistance = thinPSTransformation.L2NORM(x, c) #Get the eucledian distance
        K = thinPSTransformation.U(eucledianDistance) #Apply the bend energy Equation
        w =  t[:-3] #Ensure that sum of wi = 0
        a = t[-3:] #get a1, ax*x, and ay*y
        sumValue = np.dot(K, w) #This is the sum equation at the end of the equation
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + sumValue

'''
Create empty grid.
'''
def gridNorm(gridShape):
    height,width = gridShape[:2]
    L_1 = np.empty((height, width, 2))
    L_1[..., 0] = np.linspace(0, 1, width)
    L_1[..., 1] = np.expand_dims(np.linspace(0, 1, height, dtype=np.float32), -1)
    return L_1
    

'''
Once the grid or image is scaled correctly get the dx and dy needed for minimizing the bending energy. Apply the interpolant equation
'''
def warpGrid(t, pointsT, dimensions):
    gridNormalized = gridNorm(dimensions)
    xNorm = gridNormalized.reshape((-1, 2))
    yNorm = gridNormalized.reshape((-1, 2))
    dx = thinPSTransformation.interpoltantEquation(xNorm, pointsT, t[:, 0])
    dx = dx.reshape(dimensions[:2])
    dy = thinPSTransformation.interpoltantEquation(yNorm, pointsT, t[:, 1])
    dy = dy.reshape(dimensions[:2])
    gridWarpped = np.stack((dx, dy), -1)
    gridWarpped = gridWarpped + gridNormalized
    return gridWarpped
    
'''
get the remapped coodinates and separate them into their respective x y list. This will be used in for opencv's remap function.
'''
def remapCoordinates(gridOriginal, dimensions):
    rescaleX = (gridOriginal[:, :, 0] * dimensions[1])
    rescaleX = rescaleX.astype(np.float32)
    rescaleY = (gridOriginal[:, :, 1] * dimensions[0])
    rescaleY = rescaleY.astype(np.float32)
    return rescaleX, rescaleY
'''
get the coordinates of source and target and then calculate their change. Then calculate theta for both the x coordinates and y coordinates.
'''
def getPointsT(pointsS, pointsT):
    change = pointsS - pointsT
    points_X = np.column_stack((pointsT, change[:, 0]))
    points_Y = np.column_stack((pointsT, change[:, 1]))
    T_X = thinPSTransformation.calcTheta(points_X)
    T_Y = thinPSTransformation.calcTheta(points_Y)
    return np.stack((T_X, T_Y), -1)

'''
get the min and max values needed to create the cropped image. It will get the contours of the word and get the max and min y values and it will be used to crop the image.
'''
def coordDictionary(matrixContours):
  yCoord = []
  for row in matrixContours:
      yCoord.append(row[1])
  return min(yCoord), max(yCoord)

'''
separate the x and y list and change the target points to be the straigt line value which will be the middle of the image.
'''
def topBottomSpline(matrixContours, value, height):
  returnList = []
  for row in matrixContours:
      temp = []
      temp.append(row[0])
      temp.append(height/2)
      #temp.append(abs(value))
      returnList.append(temp)
  return returnList


'''
This has not been used yet, but will be used to make the TCL much more spread out and reduce the energy.
'''
def straighten(matrixContours):
  returnList = []
  index = 25
  for row in matrixContours:
      temp = []
      temp.append(row[0]  + index * 5)
      temp.append(row[1])
      returnList.append(temp)
      index -= 1
  return returnList


'''
Create a dictionary that contains the x coordinates and their respective y coordinates. This was created because sometimes there were multiple y coordiantes for the same y coordinates. We only need one x and y coresponding pair.
'''
def createDictionary(matrixContours):
  xCoor = []
  yCoor = []
  output = {}
  for row in matrixContours:
      if(row[0] not in output):
        output[row[0]] = []
        output[row[0]].append(row[1])
      else:
        output[row[0]].append(row[1])
  return output

'''
split the x and y list from the dictionary
'''
def splitP(outputDict):
  returnList = []
  for x in outputDict:
    if(len(outputDict[x]) == 1):
      temp = []
      temp.append(x)
      temp.append(outputDict[x][0])
      returnList.append(temp)
  return returnList

'''
combine x list and y list together
'''
def combineXY(xList,yList):
  returnList = []
  for x,y in zip(xList,yList):
      temp = []
      temp.append(x)
      temp.append(y)
      returnList.append(temp)
  return returnList

'''
calculate the max and min of a list
'''
def minMax(listInput):
  xCoor = []
  for x in listInput:
      temp = []
      xCoor.append(x[0])
  return min(xCoor), max(xCoor)


'''
This function actually warps the image it will calculate theta, then warp the grid or the image, remap the x and y values and then calls cv2.remap to actually remap the image according to how the grid is mapped.
'''
def warpImage(pointsS, pointsT, originalImg, dimensions):
    t_Final = getPointsT(pointsS, pointsT) #get theta from source points and target points
    warppedGrid = warpGrid(t_Final, pointsT, dimensions)
    xCoord, yCoord = remapCoordinates(warppedGrid, dimensions)
    warppedImage = cv2.remap(originalImg, xCoord, yCoord, cv2.INTER_CUBIC)
    return warppedImage
'''
Plots the image.
Top left is the original.
Top Right is the original with the TCL line
Bottom Left is the warpped image
Bottom Right is the warpped image with the target points.
'''
def plotwarppedImage(img, warpImg, pointsS, pointsT, path):
    cv2.imwrite(path, warpImg)
    fig, axis = plt.subplots(2, 2, figsize=(32,16))
    axis[0][0].imshow(img[...,::-1], origin='upper', label='original')
    axis[0][1].imshow(img[...,::-1], origin='upper')
    Xdata = pointsS[:, 0]*img.shape[1]
    YData = pointsS[:, 1]*img.shape[0]
    axis[0][1].scatter(Xdata, YData, color='blue', label='source Points')
    axis[1][0].imshow(warpImg[...,::-1], origin='upper')
    axis[1][1].imshow(warpImg[...,::-1], origin='upper')
    Xdata = pointsT[:, 0]*warpImg.shape[1]
    YData = pointsT[:, 1]*warpImg.shape[0]
    axis[1][1].scatter(Xdata,YData,color='red', label='post TPS')
    axis[0][0].legend()
    axis[0][1].legend()
    axis[1][0].legend()
    axis[1][1].legend()
    plt.show()
'''
transformation function
'''
def tpsTransformation(img, filename, pathToTCL):
  ims = Image.open(filename)
  w,h = ims.size
  tcl_LinesX = getFiles(pathToTCL) #Get Text Center Line
  sshape =  tcl_LinesX
  #print(sshape)
  print("WIDTH")
  print(w)
  print("HEIGHT")
  print(h)
  contours = getContours()
  minY, maxY = coordDictionary(contours)
  
  tshapeTop = topBottomSpline(sshape, maxY-h/2, h) #Get the top Spline of the word
  tshapeBottom = topBottomSpline(sshape, minY+h/2, h) #get the bottom spline of the word

  #tshapeTop[len(tshapeTop)-1][1] = tshapeTop[len(tshapeTop)-1][1]
  #tshapeBottom[len(tshapeBottom)-1][1] = tshapeBottom[len(tshapeBottom)-1][1] - 1

  #Normalize the source points and the target points
  index = 0
  listTemp = []
  for x in sshape:
    temp = []
    temp.append(x[0]/w)
    temp.append(x[1]/h)
    listTemp.append(temp)
    if (index == 30):
      break
    else:
      index += 1
  sshape = listTemp
  #print(sshape)
  listTemp = []
  index = 0
  for x in tshapeBottom:
    temp = []
    temp.append(x[0]/w)
    temp.append(x[1]/h)
    listTemp.append(temp)
    if (index == 30):
      break
    else:
      index += 1
  tshapeBottom = listTemp

  #print(tshapeBottom)
  #print("LENGTH")
  #print(len(sshape))
  #print(len(tshapeBottom))
  
  #Convert to np.arrays
  pointsS = np.array(sshape)
  pointsT = np.array(tshapeBottom)
  warped = warpImage(pointsS, pointsT, img, (h,w))
  xList = pointsT[:, 0]*warped.shape[1]
  yList = pointsT[:, 1]*warped.shape[0]
  #Used to crop image
  cdst = combineXY(xList,yList)
  
  #Plot and save image. This image is then used again for the top splie
  plotwarppedImage(img, warped,pointsS, pointsT, '/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/intermediate/final.jpg')

  img = cv2.imread('/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/intermediate/final.jpg', 1)
  ims = Image.open('/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/intermediate/final.jpg')
  w,h = ims.size
    
  #Follows same formula as Bottom Spline
  tcl_LinesX = getFiles(pathToTCL)
  sshape =  cdst
  #print(sshape)
  minY, maxY = coordDictionary(contours)
  tshapeTop = topBottomSpline(sshape, maxY-h/2, h)
  tshapeBottom = topBottomSpline(sshape, minY+h/2, h)
  #tshapeTop[len(tshapeTop)-1][1] = tshapeTop[len(tshapeTop)-1][1] - 1
  #tshapeBottom[len(tshapeBottom)-1][1] = tshapeBottom[len(tshapeBottom)-1][1] - 1

  #print(tshapeTop)
  #print(tshapeBottom)

  #Top p2
  index = 0
  listTemp = []
  for x in sshape:
    temp = []
    temp.append(x[0]/w)
    temp.append(x[1]/h)
    listTemp.append(temp)
    if (index == 30):
      break
    else:
      index += 1
  sshape = listTemp
  #print(sshape)
  listTemp = []
  index = 0
  print(tshapeTop)

  minX, maxX = minMax(tshapeTop)
  Yvalue = tshapeTop[0][1]
  
  for x in tshapeTop:
    temp = []
    temp.append(x[0]/w)
    temp.append(x[1]/h)
    listTemp.append(temp)
    if (index == 30):
      break
    else:
      index += 1
  tshapeTop = listTemp
  #print(tshapeTop)
 

  pointsS = np.array(sshape)

  pointsT = np.array(tshapeTop)

  #print(c_src)
  #print(c_dst)
  #print(type(c_src))
  #print(type(c_dst))
  warped = warpImage(pointsS, pointsT, img, (h,w))
  endpath = filename[-12:]
  endpath = '/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/finalImages/' + endpath
  plotwarppedImage(img, warped,pointsS, pointsT, endpath)
  
  if (maxY-h/2 > h):
    maxY = h - 20
  if (minY+h/2 < 0):
    minY = h + 20
    
  returnCoordCrop = [minX - 50, h/10+Yvalue, maxX+50, Yvalue-h/10]  #This is used to crop the image [left, top, right, bottom]
  return returnCoordCrop

'''
Crop your image
'''
def changeImage(imagePath, destinationPath, listCoord):
    img_dir = imagePath
    imageReturnString = imagePath[-12:]
    #print("IMAGE RETURN STEING")
    #print(imageReturnString)
    im = Image.open(imagePath)
    width, height = im.size
    #print(im.size)
    left = listCoord[0]
    if (left < 0):
      left = 1
    top = listCoord[1]
    if (top > height):
      top = height - 1
    right = listCoord[2]
    if (right > width):
      right = width - 1
    bottom = listCoord[3]
    if (bottom < 0):
      bottom = 1
    if (top > bottom):
      temp = top
      top = bottom
      bottom = temp
    if (left > right):
      temp = left
      left = right
      right = temp

    areaCropped = (left, top, right, bottom)
    print(areaCropped)
    #im.tile = [e for e in im.tile if e[1][2] < width and e[1][3]<height]

    im1 = im.crop(areaCropped)
    plt.figure()
    plt.imshow(im1)
    plt.show()

    im1.save(destinationPath + imageReturnString)
    pass
    
def main():
  path = '/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/CUTE80/CUTE80/images/image061.jpg'#63.jpg, 78.JPG
  img = cv2.imread(path, 1) #Read image as grey scale
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #img = rgb2gray(img)
  #img = lit.rgb2gray(img)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  print(path)
    #'/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/CUTE80/output/image056.jpg_TCL.txt'
  pathToTCL = str(path[-12:]) + "_TCL.txt"
  pathToTCL = '/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/CUTE80/output/' + pathToTCL
  print(pathToTCL)
  returnCoordCrop = tpsTransformation(img, path, pathToTCL)
  destinationPath = '/content/drive/My Drive/Colab Notebooks/TextSnake.pytorch/finalImages/'
  imagePath = destinationPath + str(path[-12:])
  changeImage(imagePath, destinationPath, returnCoordCrop)
if __name__ == "__main__":
  main()
