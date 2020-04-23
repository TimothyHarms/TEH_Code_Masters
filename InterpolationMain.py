from ReadWriteOperations import *
import hnswlib
import numpy as np
import scipy as sp
import sklearn.neighbors as knn

def mseCalc(C,CHat):
    """Calculate the Mean Squared Error of predicted values CHat based on ground truth YStar.
        CHat and YStar must be vectors of the same length"""
    if len(C) == len(CHat):
        C = np.asarray(C).squeeze()
        CHat = np.asarray(CHat).squeeze()
        e = np.subtract(C,CHat)
        eT = np.transpose(e)
        mse = float((np.matmul(eT,e))/(CHat.shape[0]))
        return mse
    else:
        print("Matrix dimensions of Predictor and actual values must be the same")
        pass

def KNearestMulti(X,XStar,flag,k = 8):
    """Evaluates the k-nearest neighbours in `X` of each point in `XStar` \n
        `flag` sets the searching method to be used, either: \n
        'hnsw' - heirarchical navigable small world \n
        'knn' - kd tree \n
        Returns `dist` and `ind`, the distances and indexes for every value of XStar """

    if flag == 'knn':
        nbrs = knn.NearestNeighbors(n_neighbors=k,
                                    algorithm='brute').fit(X)
        dist,ind = nbrs.kneighbors(XStar)
        # dist is the distances of k number of neighbours of XStar[i] compared to X[:], an n*-by-k matrix
        # ind is the indexes of X for each neighbour of XStar, an n*-by-k matrix
    elif flag == 'hnsw':
        N = X.shape[0]
        X_indexes = np.arange(N)
        p = hnswlib.Index(space = 'l2', dim = X.shape[1])
        p.init_index(max_elements = N, ef_construction = 200, M = 16)
        p.add_items(X, X_indexes)
        p.set_ef(50)
        ind,dist = p.knn_query(XStar,k = k)
    else:
        print("Use Valid Searching flag")
        dist = ""
        ind = ""
    return dist,ind

def EditCase(XStar,X,C,Case = '3'):
    """Returns the tendon-muscle value `Cnew` of associated centroids `X` \n
    by editing `C` based on its location in the muscle geometry. \n
    The String `Case` determines the boundaries of a central region of muscle \n
    as calculated by the MATLAB code `mode_analysis.m`"""
    Cnew = np.zeros((np.shape(X)[0],1))

    if Case in ('1','2','3','4','5','6','8','9'):   
        
        xupper3 = 0.75*max(XStar[:,0])
        xlower3 = 0.25*max(XStar[:,0])
        idx = []

        for n,x in enumerate(X[:,0]):
            if x >= xupper3 or x <= xlower3:
                idx.append(n)
            else:
                pass
        
        Cnew[idx,:] = C[idx,:]

        return Cnew
    
    elif Case == '7':
                
        xupper3 = 0.7*max(XStar[:,0])
        xlower3 = 0.3*max(XStar[:,0])
        idx = []

        for n,x in enumerate(X[:,0]):
            if x >= xupper3 or x <= xlower3:
                idx.append(n)
            else:
                pass
        
        Cnew[idx,:] = C[idx,:]

        return Cnew
    else:
        print('Use valid aponeurosis case')

def ThreeDPointInter(X,C,XStar,SFlag,IFlag,k = 8,smooth = 1):
    
    """Evaluates outputs (CHat) at Locations (XStar) based on known values X and C.
       `X` represents the 3 Dimensional spatial data as coordinates,
       `C` is a value of interest associated with that coordinate and 
        IFlag sets the flag for interpolation to be used. \n
            Valid Inputs are: \n
            'gaussian'-gaussian function \n
            'thin_plate'-thin plate spline \n
            'multiquadric' -multiquadratic \n
            'inverse'-Inverse mq \n
            'idw'-Inverse distance weighting \n
        SFlag sets the flag for searching method.\n
            Valid inputs are: \n
            'knn -k-nearest neighbours \n
            'hnsw-hierarchical navigable small world search. \n
        k sets the no. nearest neighbours or the radius of the graph searching method."""

    dist,ind = KNearestMulti(X,XStar,SFlag,k)
    CHat = np.ones(np.shape(XStar)[0])

    if IFlag in ('gaussian' 'thin_plate' 'multiquadric' 'inverse'): # Perform rbf interpolation on Xstar based on its k-nearest neighbours in X
        for i, (dist_i, ind_i) in enumerate(zip(dist, ind)):
            x = X[ind_i,0] # Generate slices of x,y,z, the coordinates of the k nearest neighbours of XStar[i]
            y = X[ind_i,1]
            z = X[ind_i,2]
            Cnn = C[ind_i]
            if len(np.unique(Cnn)) == 1: # If all values of k-nearest neighbours (Cnn) are the same, CHat takes that value. Avoids singular matrices
                CHat[i] = Cnn[0]
            else:
                rbfi = sp.interpolate.Rbf(x,y,z,Cnn,function = IFlag)
                CHat[i] = rbfi(XStar[i,0],XStar[i,1],XStar[i,2])
        return CHat

    elif IFlag == 'idw': # Perform idw interpolation on XStar based on the k-nearest neighbours in X
        for i, (dist_i, ind_i) in enumerate(zip(dist, ind)): #evaluate iteratively each row of distances and associated indexes
            Cnn = C[ind_i]
            if 0 not in dist_i:
                W_i = 1.0 / (np.power(dist_i,smooth)) #calculate weights for YHat_i's k nearest neighbours (k length array)
                CHat[i] = np.dot(W_i,C[ind_i])/np.sum(W_i)
            else:
                ZeroLocation = np.argwhere(dist_i == 0) #find the location where distance = 0 in dist_i
                IndexC = ind_i[ZeroLocation]
                if np.size(IndexC) == 1:
                    CHat[i] = C[IndexC]
                else:
                    CHat[i] = C[IndexC[0]]
        return CHat

    else:
        print('Use valid interpolation flag.')
        return ""

if __name__ == "__main__":
    # New Centroids File-----------------------------------------------------------------------------------------------
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,DataNCF = CreateMatrixDat(NCF_Str)
    XStar = DataNCF
    # New Centroids File C---------------------------------------------------------------------------------------------
    NCFC_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles","new_centroids_file_C.dat")
    HeaderNCFC,DataNCFC = CreateMatrixDat(NCFC_Str)
    CStar = DataNCFC
    # Analyzed Model---------------------------------------------------------------------------------------------------
    AM_str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","analyzed_model.dat")
    HeaderAM,DataAM = CreateMatrixDat(AM_str)
    X = DataAM[:,:3]
    C = DataAM[:,3]
    # Run interpolation ----------------------------------------------------------------------------------------------
    Cnew = EditCase(XStar,X,C)
    CHat = ThreeDPointInter(X,Cnew,XStar,'hnsw','idw')
    # Write Data-------------------------------------------------------------------------------------------------------
    header = "TITLE = \"ANALYZED MODEL\"\nVARIABLES = \"C\"\nZONE T=\"Step 0 Incr 0\"\n STRANDID=0, SOLUTIONTIME=0\n I=7553, J=1, K=1, ZONETYPE=Ordered\n DATAPACKING=POINT\n DT=(SINGLE)"

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/New_Centroids_file_C.dat",CHat,header = header,comments='')