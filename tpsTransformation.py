class thinPSTransformation:
    '''
    L2NORM 
    '''
    @staticmethod
    def L2NORM(xyi, xy):
        topLeftMatrix = xyi[:, None, :2]
        bottomLeftMatrix = xy[None, :, :2]
        squareV = np.square(topLeftMatrix - bottomLeftMatrix ).sum(-1)
        sqrtV = np.sqrt(squareV)
        return sqrtV
    '''
    Bending energy equation  
    '''
    @staticmethod
    def U(rij):
        equationResult = pow(rij,2) * np.log(rij+1e-6)
        return equationResult

    @staticmethod
    def calcTheta(point_Coord):        
        width = point_Coord.shape[0]
        print(width)
        K = thinPSTransformation.U(thinPSTransformation.L2NORM(point_Coord, point_Coord))
        P = np.ones((width, 3), dtype=np.float32)
        P[:, 1:] = point_Coord[:, :2]
        P_T = P.T
        O = np.zeros((width+3, width+3), dtype=np.float32)
        o = np.zeros(width+3, dtype=np.float32)
        o[:width] = point_Coord[:, -1]
        O[:width, :width] = K
        O[:width, -3:] = P
        O[-3:, :width] = P_T
        print(K)
        print(P)
        print(P_T)
        print(O)
        t_Final = np.linalg.lstsq(O,o)[0] # Least sqaures regression 
        return t_Final

    @staticmethod
    def interpoltantEquation(x, c, t):
        x = np.atleast_2d(x)
        eucledianDistance = thinPSTransformation.L2NORM(x, c)
        K = thinPSTransformation.U(eucledianDistance)
        w =  t[:-3]
        a = t[-3:]
        sumValue = np.dot(K, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] +sumValue


def gridNorm(gridShape):
    height,width = gridShape[:2]    
    L_1 = np.empty((height, width, 2))
    L_1[..., 0] = np.linspace(0, 1, width)
    L_1[..., 1] = np.expand_dims(np.linspace(0, 1, height, dtype=np.float32), -1)
    return L_1
    
def getPointsT(pointsS, pointsT):
    change = pointsS - pointsT
    points_X = np.column_stack((pointsT, change[:, 0]))
    points_Y = np.column_stack((pointsT, change[:, 1]))
    T_X = thinPSTransformation.calcTheta(points_X)
    T_Y = thinPSTransformation.calcTheta(points_Y)
    return np.stack((T_X, T_Y), -1)


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
  
 def warpImage(pointsS, pointsT, originalImg, dimensions):
    t_Final = getPointsT(pointsS, pointsT) #get theta from source points and target points
    warppedGrid = warpGrid(t_Final, pointsT, dimensions)
    rescale_XCoord = (warppedGrid[:, :, 0] * dimensions[1])
    rescale_XCoord = rescale_XCoord.astype(np.float32)
    rescale_YCoord = (warppedGrid[:, :, 1] * dimensions[0])
    rescale_YCoord = rescale_YCoord.astype(np.float32)
    warppedImage = cv2.remap(originalImg, rescale_XCoord, rescale_YCoord, cv2.INTER_CUBIC)
    return warppedImage

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
