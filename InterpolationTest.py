from InterpolationMain import *
from FibreGenerationMain import *
from ReadWriteOperations import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import pi

# File Containing Testing and helper functions for InterpolationMain and FibreGenerationMain.

def SparsityTest():
    fracts = np.arange(0.05,1.0,0.05)
    Methods = ['gaussian','multiquadric','inverse','thin_plate','idw']
    
    for M in Methods:
        i = 0
        mse = np.zeros(len(fracts))
        for f in fracts:
            mse[i] = ThreeDTest('hnsw',M,f)
            i += 1
        plt.plot(fracts,mse,label = M)
 
    plt.legend()
    plt.show()

def ThreeDTest(SMethod,IMethod,Fraction,Plot = False):

    """"Generates a cylinder with associated values and tests the function ThreeDMeshInter against it. \n
    `Method` is the type of interpolation used, see InterpolationMain for details \n
    `Fraction` is the proportion of data used to test the function \n
    `Plot` is a boolean that specifies whether you want the function to plot or not"""
    
    # Cylinder Parameters--------------------------------------------------------- 
    CL = 100 # cylinder length
    Pt = 120 # number of points in each cylinder
    Cn = 50 # number of horizontal slices in cylinder

    x = np.zeros(Cn*Pt)
    y = np.zeros(Cn*Pt)
    z = np.zeros(Cn*Pt)
    # Generate cylinder-----------------------------------------------------------
    n = 0
    for i in range(Cn):
        for j in range(Pt):
            x[n] = np.cos((2*pi*j)/Pt)
            y[n] = np.sin((2*pi*j)/Pt)
            z[n] = i*(CL/Cn)
            n += 1
    
    YFull = (np.sin(2*pi*0.03*z))+(np.cos(2*pi*x+2*pi*x))
    XFull = np.column_stack((x,y,z))
    MFull = np.column_stack((x,y,z,YFull))

    # Randomise matrix and Generate sparse version of geometry--------------------
    split = int(np.ceil((MFull.shape[0])*Fraction)) 
    np.random.shuffle(MFull)
    # Sparse Set
    XTrain = MFull[:split,:3]
    YTrain = MFull[:split,3]
    # Training set
    XStar = MFull[split:,:3]
    CStar = MFull[split:,3]

    # Reconstruct XFull's geometry using XTrain and YTrain------------------------
    YHat = ThreeDPointInter(XTrain,YTrain,XFull,SMethod,IMethod,10)
    mse = mseCalc(YFull,YHat)
    print('Mean Squared Error =',mse)
    # Plot whole data-----------------------------------------------------------
    if Plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(XFull[:,0],XFull[:,1],XFull[:,2],c=[float(i) for i in YFull],cmap='plasma')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        # Plot training Data
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(XTrain[:,0],XTrain[:,1],XTrain[:,2],c=[float(i) for i in YTrain],cmap='plasma')
        ax2.set_xlabel('XTrain1')
        ax2.set_ylabel('XTrain2')
        ax2.set_zlabel('XTrain3')
        # Plot Reconstruction of XFull
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(XFull[:,0],XFull[:,1],XFull[:,2],c=[float(i) for i in YHat],cmap='plasma')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        
        plt.show()

    return mse

def DisplayCentroids(Centroids,outputs,ax,N=1,sections=1):
    """Plots a slice of centroids and outputs as a 3D scatter plot \n
    `N` determines the segment of the total data to show \n
    `sections` determines how many segments to split the data in along the long axis (x values)"""

    SliceValues = np.linspace(float(min(Centroids[:,0])),float(max(Centroids[:,0])),sections+1) # Create boundaries in x for each slice.
    idx1 = np.asarray((Centroids[:,0]>=SliceValues[N-1]))*np.asarray((Centroids[:,0]<=SliceValues[N]))

    idx1 = idx1.flatten() 

    CentroidSlice = Centroids[idx1,:]
    
    outputSlice = outputs[idx1,:]

    # Plot Data-------------------------------------------------------------------------------------------------------
    ax.scatter(CentroidSlice[:,0],CentroidSlice[:,1],CentroidSlice[:,2],c = [float(N) for N in outputSlice],cmap = 'bwr')
    ax.set_zlabel('z')
    ax.set_ylabel('y')
    ax.set_xlabel('x')

def InterpolateCentroidsPure():
    # New Centroids File-----------------------------------------------------------------------------------------------
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/Project_Gastro/workflows/Cesim/musc_mod_v2/OutputFiles","new_centroids_file.dat")
    HeaderNCF,DataNCF = CreateMatrixDat(NCF_Str)
    XStar = DataNCF
    # New Centroids File C---------------------------------------------------------------------------------------------
    NCFC_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/Project_Gastro/workflows/Cesim/musc_mod_v2/OutputFiles","new_centroids_file_C.dat")
    HeaderNCFC,DataNCFC = CreateMatrixDat(NCFC_Str)
    CStar = DataNCFC
    # Analyzed Model---------------------------------------------------------------------------------------------------
    AM_str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/Project_Gastro/workflows/Cesim/musc_mod_v2/OutputFiles","analyzed_model.dat")
    HeaderAM,DataAM = CreateMatrixDat(AM_str)
    X = DataAM[:,:3]
    C = DataAM[:,3]
    # Run interpolation ----------------------------------------------------------------------------------------------
    CHat = ThreeDPointInter(X,C,XStar,'hnsw','idw')
    print(mseCalc(CStar,CHat))
    # Plot Data-------------------------------------------------------------------------------------------------------
    fig = plt.figure()
    
    ax1 = fig.add_subplot(311,projection = '3d')
    XStar2,CHat2 = SparseData(XStar,CHat,0.1)
    DisplayCentroids(XStar2,CHat2,ax1)

    ax2 = fig.add_subplot(312,projection = '3d')
    XStar3,CStar2 = SparseData(XStar,CStar,0.1)
    DisplayCentroids(XStar3,CStar2,ax2)

    ax3 = fig.add_subplot(313,projection = '3d')
    X2,C2 = SparseData(X,C,0.1)
    DisplayCentroids(X2,C2,ax3)

    plt.show()

def InterpolateCentroidsNew():
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
    print(mseCalc(CStar,CHat))
    # Plot Data-------------------------------------------------------------------------------------------------------
    fig = plt.figure()
    
    ax1 = fig.add_subplot(311,projection = '3d')
    XStar2,CHat2 = SparseData(XStar,CHat,0.1)
    DisplayCentroids(XStar2,CHat2,ax1)

    ax2 = fig.add_subplot(312,projection = '3d')
    XStar3,CStar2 = SparseData(XStar,CStar,0.1)
    DisplayCentroids(XStar3,CStar2,ax2)

    ax3 = fig.add_subplot(313,projection = '3d')
    X2,C2 = SparseData(X,C,0.1)
    DisplayCentroids(X2,C2,ax3)

    plt.show()

    header = "TITLE = \"ANALYZED MODEL\"\nVARIABLES = \"C\" \nZONE T=\"Step 0 Incr 0\" \n STRANDID=0, SOLUTIONTIME=0\n I=7553, J=1, K=1, ZONETYPE=Ordered\n DATAPACKING=POINT\n DT=(SINGLE)"

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/New_Centroids_file_C_Python.dat",CHat,header = header,comments='')

# FibreGeneration Testing and Helper Functions

def DisplayMesh():
    """Simple plotter for triangular mesh"""
    
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, Vertices, Triangles = CreateMatrixVTK(VTKString)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection = '3d')
    ax1.plot_trisurf(Vertices[:,0],Vertices[:,1],Vertices[:,2],triangles= Triangles[:,1:])
    ax1.set_zlabel('z')
    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    plt.show()

def InterpolateSurfaceVectors():
    """calculate the vector field of a simple surface mesh and map it \n
    onto a volumetric centroid using InterpolationMain.py"""
    
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    
    # Interpolate Vectors
    Vectors2 = VectorInter(Centroids1,Vectors1,Centroids2)
    # Make the data sparser to display better.
    C1,V1 = SparseData(Centroids1,Vectors1,0.2)
    C2,V2 = SparseData(Centroids2,Vectors2,0.2)

    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(121,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,1,10)

    ax2 = fig.add_subplot(122,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,1,10)

    plt.show()

    header = 'TITLE = \"Normal Surface Vectors\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/NormalVectorInterpolation.dat",Vectors2,header = header,comments='')

def InterpolateSurfaceOrtho():
    """calculate the vector field of a simple surface mesh and map it \n
    orthogonally onto a volumetric centroid using InterpolationMain.py"""
   
    print('Opening Data...')
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    
    # Rotate Vectors
    RotVectors1 = LongaxisOrtho(Vectors1)
    print('Vectors Rotated \n Interpolating Centroids...')
    
    # Interpolate Vectors
    Vectors2 = VectorInter(Centroids1,RotVectors1,Centroids2)
    # Make the data more sparse to display better.
    C1,V1 = SparseData(Centroids1,RotVectors1,0.1)
    C2,V2 = SparseData(Centroids2,Vectors2,0.1)
    print('Interpolation Finished \n Plotting...')
    
    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(211,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,7,10)

    ax2 = fig.add_subplot(212,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,7,10)

    plt.show()

    header = 'TITLE = \"Normal Surface Vectors\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/OrthoVectorInterpolation.dat",Vectors2,header = header,comments='')

def InterpolateSurfaceVectorsWithPlane():
    """calculate the vector field of a surface mesh plus a central axis plane and map it \n
    onto a volumetric centroid using InterpolationMain.py"""
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    print('Loading Finished \n Inserting Plane...')
    # Create Plane of vectors through centreline.
    PlaneCentroids,PlaneVectors = InsertPlane(Centroids1,Vectors1,50,8)
    print('Plane Inserted \n Interpolating Centroids...')
    # Interpolate Vectors
    Vectors2 = VectorInter(PlaneCentroids,PlaneVectors,Centroids2)
    # Make the data more sparse to display better.
    C1,V1 = SparseData(PlaneCentroids,PlaneVectors,0.5)
    C2,V2 = SparseData(Centroids2,Vectors2,0.5)
    print('Interpolation Finished \n Plotting...')
    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(121,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,5,10)

    ax2 = fig.add_subplot(122,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,5,10)

    plt.show()

    header = 'TITLE = \"Normal Surface Vectors With Central axis Plane\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/SurfacePlaneVectorInterpolation.dat",Vectors2,header = header,comments='')

def InterpolateSurfaceVectorsWithLine():
    """calculate the vector field of a surface mesh plus a central axis line and map it \n
    onto a volumetric centroid using InterpolationMain.py"""
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    print('Loading Finished \n Inserting Centre Line...')
    # Create Plane of vectors through centreline.
    PlaneCentroids,PlaneVectors = InsertCentreLine(Centroids1,Vectors1,50)
    print('Centre Line Inserted \n Interpolating Centroids...')
    # Interpolate Vectors
    Vectors2 = VectorInter(PlaneCentroids,PlaneVectors,Centroids2)
    # Make the data more sparse to display better.
    C1,V1 = SparseData(PlaneCentroids,PlaneVectors,0.1)
    C2,V2 = SparseData(Centroids2,Vectors2,0.1)
    print('Interpolation Finished \n Plotting...')
    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(121,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,5,10)

    ax2 = fig.add_subplot(122,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,5,10)

    plt.show()

    header = 'TITLE = \"Normal Surface Vectors With Central axis Line\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/SurfaceLineVectorInterpolation.dat",Vectors2,header = header,comments='')

def CreateBiPennate1():
    """calculate the vector field of a surface mesh with angled fibres towards the central x axis \n
    plus a central axis plane and map it onto a volumetric centroid using InterpolationMain.py"""
    
    print('Opening Data...')
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    print('Loading Finished \n Rotating Vectors...')
    
    # Rotate Vectors
    RotVectors1 = np.zeros((Vectors1.shape[0],3))

    idxpos = np.argwhere(Centroids1[:,1] >= 0)
    idxpos = idxpos.flatten()
    idxneg = np.argwhere(Centroids1[:,1] < 0)
    idxneg = idxneg.flatten()

    PosVectors = RotationTransform(Vectors1[idxpos,:],degZ = 30)
    NegVectors = RotationTransform(Vectors1[idxneg,:],degZ = -30)
    RotVectors1[idxpos,:] = PosVectors[:,:]
    RotVectors1[idxneg,:] = NegVectors[:,:]
    print('Vectors Rotated \n Inserting Plane...')
    
    # Create Plane of vectors through centreline.
    PlaneCentroids,PlaneVectors = InsertPlane(Centroids1,RotVectors1,50,4)
    print('Plane Inserted \n Interpolating Centroids...')
    
    # Interpolate Vectors
    Vectors2 = VectorInter(PlaneCentroids,PlaneVectors,Centroids2)
    # Make the data more sparse to display better.
    C1,V1 = SparseData(PlaneCentroids,PlaneVectors,1)
    C2,V2 = SparseData(Centroids2,Vectors2,1)
    print('Interpolation Finished \n Plotting...')
    
    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(121,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,5,10)

    ax2 = fig.add_subplot(122,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,5,10)

    plt.show()

    header = 'TITLE = \"New Centroid Vectors\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/BiPennateCentralPlaneFibres30.dat",Vectors2,header = header,comments='')

def CreateBiPennate2():
    """calculate the vector field of a surface mesh with angled fibres towards a central x-y plane \n
    plus a central axis plane and map it onto a volumetric centroid using InterpolationMain.py"""
    
    print('Opening Data...')
    # Load Surface Mesh Data and generate normals
    VTKString = OpenData('C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/InputFiles','muscle_surface.vtk')
    header, PointData, PolygonData = CreateMatrixVTK(VTKString)
    Centroids1,Vectors1 = ElementNormal(PointData,PolygonData)
    Vectors1 = LongaxisOrtho(Vectors1)
    # Load full volume centroid
    NCF_Str = OpenData("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/Project_Gastro/workflows/Cesim/musc_mod_v2/OutputFiles","new_centroids_file.dat")
    HeaderNCF,Centroids2 = CreateMatrixDat(NCF_Str)
    print('Loading Finished \n Rotating Vectors...')
    
    # Rotate Vectors
    RotVectors1 = np.zeros((np.shape(Vectors1)[0],3))

    idxpos = np.argwhere(Centroids1[:,1] >= 0)
    idxpos = idxpos.flatten()
    idxneg = np.argwhere(Centroids1[:,1] < 0)
    idxneg = idxneg.flatten()

    PosVectors = RotationTransform(Vectors1[idxpos,:],degZ = -30)
    NegVectors = RotationTransform(Vectors1[idxneg,:],degZ = 30)
    RotVectors1[idxpos,:] = PosVectors[:,:]
    RotVectors1[idxneg,:] = NegVectors[:,:]
    print('Vectors Rotated \n Inserting Plane...')
    
    # Create Plane of vectors through centreline.
    PlaneCentroids,PlaneVectors = InsertPlane(Centroids1,RotVectors1,50,4)
    print('Plane Inserted \n Interpolating Centroids...')
    
    # Interpolate Vectors
    Vectors2 = VectorInter(PlaneCentroids,PlaneVectors,Centroids2)
    # Make the data more sparse to display better.
    C1,V1 = SparseData(PlaneCentroids,PlaneVectors,0.1)
    C2,V2 = SparseData(Centroids2,Vectors2,0.1)
    print('Interpolation Finished \n Plotting...')
    
    # Plot Data
    fig = plt.figure()

    ax1 = fig.add_subplot(211,projection = '3d')
    DisplaySliceVectors(C1,V1,ax1,1,1)

    ax2 = fig.add_subplot(212,projection = '3d')
    DisplaySliceVectors(C2,V2,ax2,1,1)

    plt.show()

    header = 'TITLE = \"New Centroid Vectors\"\nVARIABLES = \"XV\", \"YV\", \"ZV\" \nZONE T=\"Step 0 Incr 0\" \nF = VECTORS'

    np.savetxt("C:/Users/Tim/Documents/University/Year 4/Final Project/FinalYearProjectCode/TEH_Code/OutputFiles/BiPennateCentralPlaneFibres.dat",Vectors2,header = header,comments='')

def CreateUnipenneate():
    pass

def DisplaySliceVectors(Centroids,Vectors,ax,N = 1,sections = 1):
    """Plots a slice of centroids and vectors as a 3D vector field"""

    SliceValues = np.linspace(float(min(Centroids[:,0])),float(max(Centroids[:,0])),sections+1) # Create boundaries in x for each slice.
    idx1 = np.asarray((Centroids[:,0]>=SliceValues[N-1]))*np.asarray((Centroids[:,0]<=SliceValues[N]))

    idx1 = idx1.flatten() 

    CentroidSlice = Centroids[idx1,:]
    
    VectorSlice = Vectors[idx1,:]

    # Plot Data-------------------------------------------------------------------------------------------------------
    ax.quiver(CentroidSlice[:,0],CentroidSlice[:,1],CentroidSlice[:,2],VectorSlice[:,0],VectorSlice[:,1],VectorSlice[:,2])
    ax.set_zlabel('z')
    ax.set_ylabel('y')
    ax.set_xlabel('x')

def SparseData(Input,Output,Fraction):
    """ Takes input output data and randomly samples a proportion of it. \n
    used to aid computation when plotting large datasets"""
    # Take a fraction of the full data
    MFull = np.column_stack((Input,Output))
    np.random.shuffle(MFull)
    split = int(np.ceil((MFull.shape[0])*Fraction)) 
    # Generate a Sparse Set
    C = MFull[:split,:np.shape(Input)[1]]
    V = MFull[:split,np.shape(Input)[1]:]
    return C,V

if __name__ == "__main__":
    InterpolateCentroidsNew()