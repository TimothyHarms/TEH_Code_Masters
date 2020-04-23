from pathlib import Path
import numpy as np

def OpenData(DatFolder,filename):

	"""Returns `DatString`, the string of a file\n 
	`DatFolder` - the file location.\n
	`filename` - the name of the file."""

	# The source file 
	DatFolderPath = Path(DatFolder)
	DatFile = DatFolderPath / filename
	Dat = open(DatFile)
	DatString = Dat.read()

	return DatString

def CreateMatrixDat(String):

	"""Returns two lists, `Header` , `AllData` based on `String`, a dat file."""

	# split string into lines
	String_ls = String.splitlines()

	Header = []
	AllData = []

	for Row in String_ls:
		RowDataLis = Row.split() # Create list of rowdata (Strings)
		RowDataFloat = np.zeros(len(RowDataLis))
		try: #Attempts to covert each element in row into floats
			for i,x in enumerate(RowDataLis):
				RowDataFloat[i] = float(x)
			AllData.append(RowDataFloat)
		except ValueError: #If it cannot convert an element in the row into a float, the entire row is placed into "Header"
			Header.append(Row)

	AllDataNew = np.asmatrix(AllData)

	return Header,AllDataNew

def CreateMatrixVTK(String):
    """Converts the String of a vtk file into numpy matrices"""
    # split string into lines
    surface_ls = String.splitlines()

    # create variables for first 4 lines involving information of vtk file    
    Header = surface_ls[:5]

    # Create empty variables for population
    PointDataList = []
    PolygonDataList = []

    # Define variables including 'switches' for which bit of the file the code is looking at
    n = 4
    is_pointdata = False
    is_polydata = False

    for data in surface_ls[4:]:
        if surface_ls[n-1].count("POINTS") == 1: # Change 'switch' for points data, polygon data, or neither
            is_pointdata = True
            is_polydata = False
        if surface_ls[n-1].count('POLYGONS') ==1:
            is_pointdata = False
            is_polydata = True
        if surface_ls[n].count("POINTS") == 1:
            is_pointdata = False
            is_polydata = False
        if surface_ls[n].count('POLYGONS') == 1:
            is_pointdata = False
            is_polydata = False
        if surface_ls[n].count('CELL_DATA') == 1 or surface_ls[n].count('POINT_DATA') == 1 or not surface_ls[n]:
            is_pointdata = False
            is_polydata = False

        if is_pointdata: # Populate points data based on conditional test
            PointDataList = PointDataList + data.split(' ')
        if is_polydata: # Populate polygon data based on conditional test
            PolygonDataList = PolygonDataList + data.split(' ')
    
        n = n + 1

    # Fix, remove empty points, and generate matrices -------------------------------------
    for i,val in enumerate(PointDataList) :# removes empty points from the data
        if not bool(val):
            PointDataList.pop(i)

    PointData = [float(i) for i in PointDataList] # converts data to floats

    PointData = np.array(PointData)
    rows = int(len(PointData)/3)
    PointDataMatrix = np.reshape(PointData, (rows,int(3)))

    for i,val in enumerate(PolygonDataList) :# removes empty points from the data
        if not bool(val):
            PolygonDataList.pop(i)

    PolygonData = [int(i) for i in PolygonDataList] # converts data to floats

    PolygonData = np.array(PolygonData)
    rows = int(len(PolygonData)/4)
    PolygonDataMatrix = np.reshape(PolygonData, (rows,int(4)))

    return Header, PointDataMatrix, PolygonDataMatrix

if __name__ == '__main__':
    pass