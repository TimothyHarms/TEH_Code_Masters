from InterpolationMain import ThreeDPointInter
import numpy as np
from sklearn import preprocessing


def ElementNormal(PointData,PolygonData):
    """Returns two matrices, `Centroids` and `Vectors` based on \n 
    the vertices `PointData` and the triangulation data `PolygonData` of an FE mesh\n
    `Centroids` is an N-by-3 matrix of each polygon's geometric centre \n
    `Vectors` are the normal vectors for each face of the FE mesh."""

    Centroids = np.zeros((np.shape(PolygonData)[0],3))
    Vectors = np.zeros((np.shape(PolygonData)[0],3))

    for n,Verts in enumerate(PolygonData[:,1:]): # Each element in axis 0 of PolygonData should be a set of indices referencing PointData
        TriPoints = PointData[Verts]
        # Define Vectors of two triangle edges and find normal vector
        V1 = np.subtract(TriPoints[1],TriPoints[0])
        V2 = np.subtract(TriPoints[2],TriPoints[0])

        norm = np.cross(V1,V2)
        Vectors[n,:] = norm
        
        # Find centroid of triangle
        x_midpt = np.sum(TriPoints[:,0]) / 3
        y_midpt = np.sum(TriPoints[:,1]) / 3
        z_midpt = np.sum(TriPoints[:,2]) / 3

        midpt = np.matrix([x_midpt,y_midpt,z_midpt])
        Centroids[n,:] = midpt

    VectorNorm = preprocessing.normalize(Vectors)
    return Centroids,VectorNorm 

def LongaxisOrtho(Vectors):
    """Takes a field of vectors, and returns a field of each vector \n
    rotated 90 degrees pointing along the x-axis of the overall coordinate space."""

    newVectors = np.zeros((Vectors.shape[0],3))

    for n,x in enumerate(Vectors):
        y = np.matrix([x[0],-x[2],x[1]])
        vOut = np.cross(x,y)
        newVectors[n,:] = vOut
    return newVectors

def VectorInter(Centroids1,Vectors1,Centroids2):
    """Maps a set of vectors onto the points in `Centroids2` via \n
    inverse distance weighted interpolation of points `Centroids1` with associated vectors `Vectors1`"""

    Vx = ThreeDPointInter(Centroids1,Vectors1[:,0],Centroids2,'hnsw','idw',smooth = 3.5)
    Vy = ThreeDPointInter(Centroids1,Vectors1[:,1],Centroids2,'hnsw','idw',smooth = 3.5)
    Vz = ThreeDPointInter(Centroids1,Vectors1[:,2],Centroids2,'hnsw','idw',smooth = 3.5)

    Vectors2 = np.column_stack((Vx,Vy,Vz))
    Vectors2 = preprocessing.normalize(Vectors2)
    return Vectors2

def InsertPlane(Centroids,Vectors,Nx,Nz):
    """Insert a Plane of Vectors oriented along the long axis"""

    # Generate a new set of Centroids to append to the existing data
    UniqueX = np.linspace(min(Centroids[:,0]),max(Centroids[:,0]),Nx)
    UniqueZ = np.linspace(min(Centroids[:,2]),max(Centroids[:,2]),Nz)
    
    XGrid,ZGrid = np.meshgrid(UniqueX,UniqueZ)
    
    ArrayX = np.reshape(XGrid,(Nx*Nz,1))
    ArrayY = np.zeros((Nx*Nz,1))
    ArrayZ = np.reshape(ZGrid,(Nx*Nz,1))
    newCentroids = np.column_stack((ArrayX,ArrayY,ArrayZ))
    
    # generate a new set of vectors to associate with the plane
    newVectors = np.column_stack((np.ones((Nx*Nz,1)),np.zeros((Nx*Nz,2))))

    # Append new data to existing data
    CentroidsOut = np.concatenate((Centroids,newCentroids),axis = 0)
    VectorsOut = np.concatenate((Vectors,newVectors),axis = 0)

    return CentroidsOut,VectorsOut

def InsertCentreLine(Centroids,Vectors,N):
    
    X = np.linspace(min(Centroids[:,0]),max(Centroids[:,0]),N)
    ArrayX = np.reshape(X,(N,1))
    ArrayY = np.zeros((N,1))
    ArrayZ = np.zeros((N,1))
    newCentroids = np.column_stack((ArrayX,ArrayY,ArrayZ))
    
    newVectors = np.column_stack((np.ones((N,1)),np.zeros((N,2))))

    # Append new data to existing data
    CentroidsOut = np.concatenate((Centroids,newCentroids),axis = 0)
    VectorsOut = np.concatenate((Vectors,newVectors),axis = 0)

    return CentroidsOut,VectorsOut

def RotationTransform(Vectors,degX = 0,degY = 0,degZ = 0):
    newVectors = np.zeros((Vectors.shape[0],3))
    
    pi = np.pi
    ThetaX = (2*pi*degX)/360
    ThetaY = (2*pi*degY)/360
    ThetaZ = (2*pi*degZ)/360

    RotateX = np.matrix([[1,0,0],[0,np.cos(ThetaX),-np.sin(ThetaX)],[0,np.sin(ThetaX),np.cos(ThetaX)]])
    RotateY = np.matrix([[np.cos(ThetaY),0,np.sin(ThetaY)],[0,1,0],[-np.sin(ThetaY),0,np.cos(ThetaY)]])
    RotateZ = np.matrix([[np.cos(ThetaZ),-np.sin(ThetaZ),0],[np.sin(ThetaZ),np.cos(ThetaZ),0],[0,0,1]])

    VectorsT = np.transpose(Vectors)
    newVectors = RotateX.dot(RotateY).dot(RotateZ).dot(VectorsT)

    newVectors = np.transpose(newVectors)
    return newVectors
        
if __name__ == '__main__':
    pass