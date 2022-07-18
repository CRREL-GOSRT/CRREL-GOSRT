import vtk
import numpy as np
import sys

"""Contains stand alone helper functions to facilitate the photon tracking RT code and interaction with rays and 3D mesh elements.
   Some functions here are a little old, but all of them support the photon tracking algorithms by performing vector operations and working with
   3D meshes."""

def addLine(renderer, p1, p2, color=[0.0, 0.0, 1.0], lineWidth=4.0):
    """
        Draws a point on an existing renderer connecting p1 and p2

        Inputs:
            - renderer (vtk renderer object)
            - p1 (tuple vector) coordinates of point1
            - p2 (tuple vector) coordinates of point2 - connected point

            - color (optional : 3D vector) RGB color values of point
            - lineWidth (optional : float) width of line

        Returns:
            None
    """
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(lineWidth)

    renderer.AddActor(actor)


def addPoint(renderer, p, radius=1.0, color=[0.0, 0.0, 0.0], opacity=1.0):
    """
        Draws a point on an existing renderer

        Inputs:
            - renderer (vtk renderer object)
            - p (tuple vector) coordinates of point

            - radius (optional : float) size of point
            - color (optional : 3D vector) RGB color values of point
            - opacity (optional :float 0-1) opacity of point object.

        Returns:
            None
    """
    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    renderer.AddActor(actor)


def HenYey(PHI,g=0.847):
    """Henyey-Greenstein Phase function approximation"""
    COSPHI=np.cos(PHI)
    P=(1.-g**2.)/((1.+g**2.-2.*g*COSPHI)**(3./2.))
    return P

def vecMag(v):
    """Computes 3D vector magnitude"""
    # Calculate vector magnitude
    v_mag = np.sqrt( v[0]**2 + v[1]**2 + v[2]**2 )

    return v_mag

def pts2unitVec(p1, p2):
    """Performs vector subtraction between 2 vectors, and converts to a unit vector"""
    # Calculate vector pointing from p1->p2
    v = np.array(p2) - np.array(p1)

    # Calculate unit vector
    v_unit = v / vecMag(v)
    return v_unit

def ptsDistance(p1, p2):
    """Performs vector subtraction between 2 vectors and computes distance between points"""
    # Calculate vector pointing from p1->p2
    v = np.array(p2) - np.array(p1)

    return vecMag(v)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def Fresnel(n1, n2, v_i, v_n,polar=0):
    """
    This function determines the new reflected/transmitted directions of a ray at a boundary, and their corresponding weights.

    Inputs:
        n1 -- incident medium's index of refraction
        n2 -- transmission medium's index of refraction
        v_i -- incident ray direction vector
        v_n -- surface normal direction vector
        polar -- if polar is a complex number, it will determine the photons polarization from it's imaginary part:
            1 = Horizontal polarization
            otherwise = Vertical polarization


    Returns:
        v_i_ref - (3D vector) reflected vector direction
        v_i_tran - (3D vector) transmitted vector direction
        reflectance - (float) fraction of radiation reflected
        transmitted - (float) fraction of radiation transmitted

    """

    # Incident ray directional sines and cosines
    cosThetai =  -np.dot(v_i,v_n)

    Thetai=np.arccos(cosThetai)

    sin2Thetat = (n1/n2)**2 * (1. - cosThetai**2)

    # If incident ray angle is sufficiently small, then enter Monte Carlo determination
    # of whether photon is reflected or transmitted
    # ---------------------------------------------
    #print(cosThetai,np.arccos(cosThetai),sin2Thetat)
    if n1 > n2:
        sin2ThetatCrit=sin2Thetat
    else:
        sin2ThetatCrit=0.0

    if sin2ThetatCrit <= 1.0:
        TIR=0
        # Transmitted ray directional cosine
        cosThetat = np.sqrt(1. - sin2Thetat)
        sini = np.sqrt(1. - cosThetai**2.)
        cosT = cosThetat #np.sqrt(max(0., 1. - sin2Thetat))
        cosi = cosThetai
        # Calculate reflectance:
        # ----------------------
        # First calculate the reflectance for both polarizations using Fresnel equations, then use the
        # average reflectance

        ## From Stamnes and Stamnes, RTE in coupled systems
        #Rs = ((n2 * cosi) - (n1 * cosT)) / ((n2 * cosi) + (n1 * cosT))
        #Rp = ((n1 * cosi) - (n2 * cosT)) / ((n1 * cosi) + (n2 * cosT))

        ## From http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        #Rp = ( ( n1 * cosThetai - n2 * cosThetat ) / ( n1 * cosThetai + n2 * cosThetat ) )**2
        #Rs = ( ( n2 * cosThetai - n1 * cosThetat ) / ( n2 * cosThetai + n1 * cosThetat ) )**2 ## I Think this is wrong?
        #r_parallel      = ( ( n1 * cosThetat - n2 * cosThetai ) / ( n1 * cosThetat + n2 * cosThetai ) )**2

        ## From Wiki: https://en.wikipedia.org/wiki/Fresnel_equations
        #Rs = ((n1 * cosi) - (n2 * cosT)) / ((n1 * cosi) + (n2 * cosT))
        #Rp = ((n1 * cosT) - (n2 * cosi)) / ((n1 * cosT) + (n2 * cosi))

        ## From Wiki: NO Transmission angle needed! ##
        Rs=(n1*np.cos(Thetai)-n2*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.))/(n1*np.cos(Thetai)+n2*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.))
        Rp=(n1*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.)-n2*np.cos(Thetai))/(n1*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.)+n2*np.cos(Thetai))

        ## added a special contingency that allows for polarized light stored in optional argument as a complex number.
        ## By default this number is NOT complex and photon is assumed to be unpolarized, and the reflectance follows unpolarized
        ## averaging of the Fresnel equations.
        ## if it is complex, use the according reflectance.
        if np.iscomplex(polar) == True:
            if polar.imag == 1:
                ## if imaginary part of "polar" == 1 then assume polarization is parallel (Rs)
                reflectance=Rs**2.
            else: ##otherwise, assume the polarization is perpendicular (Rp)
                reflectance=Rp**2.
        else: ## If it's not complex, return unpolarized reflectance
            reflectance = (Rs ** 2.+ Rp ** 2.) / 2.

        #print("ref",reflectance,cosi,cosT,Rs,Rp,cosThetai)
        # Reflect photon:
        # ---------------
        ##Do a check and make sure that the reflectance is okay.
        if 0 <= reflectance <= 1:
            pass
        else:
            print("Bad reflectance %.4f"%reflectance)
        v_i_ref = v_i + 2 * cosThetai * v_n
        v_i_tran = n1 / n2 * v_i + ( n1 / n2 * cosi - cosT ) * v_n

    else:

        v_i_ref = v_i + 2 * cosThetai * v_n
        reflectance=1.0
        v_i_tran=np.array([0.1,0.1,0.1])

    return v_i_ref, v_i_tran, reflectance,1.0-reflectance

def isReflected(n1, n2, v_i, v_n,polar=0,TIR=0.0):
    """
    This function determines whether the photon is reflected or transmitted

    Inputs:
        n1 - incident medium's index of refraction
        n2 - transmission medium's index of refraction
        v_i - incident ray direction vector
        v_n - surface normal direction vector
        polar - (optional) allows for reflection/refraction to have polarity
        TIR - (optional) counts total number of total-internal-reflections, helpful for killing trapped photons

    Returns:
        v_i_new - (3D vector) the new incident ray direction that results from a transmission or reflection
        reflected - (bool) flag indicating whether or not the photon was reflected
        TIR - (int) new total internal reflection count.
    """

    # Incident ray directional sines and cosines
    cosThetai =  -np.dot(v_i,v_n)

    Thetai=np.arccos(cosThetai)

    sin2Thetat = (n1/n2)**2 * (1. - cosThetai**2)

    # If incident ray angle is sufficiently small, then enter Monte Carlo determination
    # of whether photon is reflected or transmitted
    # ---------------------------------------------
    #print(cosThetai,np.arccos(cosThetai),sin2Thetat)
    if n1 > n2:
        sin2ThetatCrit=sin2Thetat
    else:
        sin2ThetatCrit=0.0

    if sin2ThetatCrit <= 1.0:
        TIR=0
        # Transmitted ray directional cosine
        cosThetat = np.sqrt(1. - sin2Thetat)
        sini = np.sqrt(1. - cosThetai**2.)
        cosT = cosThetat #np.sqrt(max(0., 1. - sin2Thetat))
        cosi = cosThetai
        # Calculate reflectance:
        # ----------------------
        # First calculate the reflectance for both polarizations using Fresnel equations, then use the
        # average reflectance

        ## From Stamnes and Stamnes, RTE in coupled systems
        #Rs = ((n2 * cosi) - (n1 * cosT)) / ((n2 * cosi) + (n1 * cosT))
        #Rp = ((n1 * cosi) - (n2 * cosT)) / ((n1 * cosi) + (n2 * cosT))

        ## From http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        #Rp = ( ( n1 * cosThetai - n2 * cosThetat ) / ( n1 * cosThetai + n2 * cosThetat ) )**2
        #Rs = ( ( n2 * cosThetai - n1 * cosThetat ) / ( n2 * cosThetai + n1 * cosThetat ) )**2 ## I Think this is wrong?
        #r_parallel      = ( ( n1 * cosThetat - n2 * cosThetai ) / ( n1 * cosThetat + n2 * cosThetai ) )**2

        ## From Wiki: https://en.wikipedia.org/wiki/Fresnel_equations
        #Rs = ((n1 * cosi) - (n2 * cosT)) / ((n1 * cosi) + (n2 * cosT))
        #Rp = ((n1 * cosT) - (n2 * cosi)) / ((n1 * cosT) + (n2 * cosi))

        ## From Wiki: NO Transmission angle needed! ##
        Rs=(n1*np.cos(Thetai)-n2*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.))/(n1*np.cos(Thetai)+n2*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.))
        Rp=(n1*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.)-n2*np.cos(Thetai))/(n1*np.sqrt(1.-(n1/n2*np.sin(Thetai))**2.)+n2*np.cos(Thetai))

        ## added a special contingency that allows for polarized light stored in optional argument as a complex number.
        ## By default this number is NOT complex and photon is assumed to be unpolarized, and the reflectance follows unpolarized
        ## averaging of the Fresnel equations.
        ## if it is complex, use the according reflectance.
        if np.iscomplex(polar) == True:
            if polar.imag == 1:
                ## if imaginary part of "polar" == 1 then assume polarization is parallel (Rs)
                reflectance=Rs**2.
            else: ##otherwise, assume the polarization is perpendicular (Rp)
                reflectance=Rp**2.
        else:
            reflectance = (Rs ** 2.+ Rp ** 2.) / 2.

        # Reflect photon:
        # ---------------
        ##Do a check and make sure that the reflectance is okay.
        if 0 <= reflectance <= 1:
            pass
        else:
            print("Bad reflectance %.4f"%reflectance)
        if np.random.random() <= reflectance:
            # Calculate reflection vector
            v_i_new = v_i + 2 * cosThetai * v_n
            reflected = True

        # Transmit photon:
        # ----------------
        else:
            # Calculate transmission vector
            v_i_new = n1 / n2 * v_i + ( n1 / n2 * cosi - cosT ) * v_n
            reflected = False

            if np.sqrt(np.sum([i**2. for i in v_i_new])) > 1.0001:
                ## Sometimes, math just doesn't work.
                print("Transmission BAD UNIT VECTOR!",v_i_new)

    # Total internal reflectance condition:
    # -------------------------------------
    else:
        # Calculate reflection vector
        TIR+=1
        v_i_new = v_i + 2 * cosThetai * v_n
        reflected = True

        if np.sqrt(np.sum([i**2. for i in v_i_new])) > 1.0001:
            ## Sometimes, math just doesn't work.
            print("TIR BAD UNIT VECTOR!",v_i_new)

    return v_i_new, reflected,TIR


def castRayAll(pSource,pTarget,obbTree,normalsMesh):
    """
    This casts a ray and returns ALL intersections along the line. Used in TracktoAbsStraight function, so
    generally not called as part of the larger RTM framework.

    Inputs:
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object

    Returns:
        distances - (array[numIntersections]) distances between each intersection
        normals - (array[numIntersections]) normal vectors at each intersection
        isHit - (bool) True/False flag indicating whether or not an intersection was found
    """

    # Check for intersection with line
    pointsIntersection = vtk.vtkPoints()   # Contains intersection point coordinates
    cellIds = vtk.vtkIdList()              # Contains intersection cell ID

    # Perform intersection test
    drct = pts2unitVec(pSource, pTarget)
    code = obbTree.IntersectWithLine(pSource, pTarget, pointsIntersection, cellIds)

    numIntersections = pointsIntersection.GetNumberOfPoints()

    distances=[]
    normals=[]
    intersections =[]
    if code == 0:
        isHit = False
        # Return empty variables
        intersectionPt = np.array([])
        cellIdIntersection = []
        normalMeshIntersection = []
    else:
        p1=pSource[:]
        isHit = True
        pointsIntersectionData = pointsIntersection.GetData()
        v_i = pts2unitVec(pSource, pTarget).squeeze()
        for idx in range(numIntersections):
            ##ensure direction is good!
            checkDrct = pts2unitVec(p1, pointsIntersectionData.GetTuple3(idx))

            v_n = np.array(normalsMesh.GetTuple(cellIds.GetId(idx)))
            intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
            cellIdIntersection = cellIds.GetId(idx)
            normalMeshIntersection = v_n

            distances.append(ptsDistance(p1,intersectionPt))
            normals.append(np.dot(v_i,v_n))
            intersections.append(intersectionPt)

            p1=intersectionPt[:]

    return np.array(distances),np.array(normals), isHit,intersections

def castFirst(pSource,pTarget,obbTree,normalsMesh):
    """
    This casts a ray and returns first intersection, used for tracking extinction coefficient.

    Inputs:
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object

    Returns:
        intersectionPt - (3D vector) location of intersection
        cellIdIntersection - (int) id of mesh cell where intersection occurs
        normalMeshIntersection - (3D vector) normal vector of intersected cell
        isHit - (bool) True/False flag indicating whether or not an intersection was found
    """

    ## Lots of care needs to be taken to avoid some funny business that sometimes occurs due to
    ## interactions between the under-the-hood algorithms that search for intersections and minor
    ## inaccuracies in the mesh.
    ## Lots of minor issues with initial intersections when occurring on an inside
    ## boundary, difficult to sort out for all meshes, and likely has to do with grain segmentation
    ## Some "hacky" fixes are used here to mitigate these issues.

    # Check for intersection with line
    pointsIntersection = vtk.vtkPoints()   # Contains intersection point coordinates
    cellIds = vtk.vtkIdList()              # Contains intersection cell ID

    # Perform intersection test
    code = obbTree.IntersectWithLine(pSource, pTarget, pointsIntersection, cellIds)
    # Return empty variables
    intersectionPt = np.array([])
    cellIdIntersection = []
    normalMeshIntersection = []
    isHit = False
    v_i=pts2unitVec(pSource,pTarget)
    if code == 0 or pointsIntersection.GetNumberOfPoints() < 1: # avoids boundaries.
        isHit = False
    else:
        pointsIntersectionData = pointsIntersection.GetData()
        for idx in np.arange(pointsIntersection.GetNumberOfPoints()):
            intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
            if ptsDistance(intersectionPt,pSource) > 0.02:
                isHit = True
                v_n = np.array(normalsMesh.GetTuple(cellIds.GetId(idx)))
                cellIdIntersection = cellIds.GetId(idx)
                intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                normalMeshIntersection = v_n
                break
                # if np.dot(v_i,v_n) > 0 and idx == 0:
                #     continue
                #
                # if idx > 3:
                #     ## TOO FAR! ##
                #     isHit = False
                #     break
                # break
            else:
                continue


    return intersectionPt, cellIdIntersection, normalMeshIntersection, isHit

def castRay(pSource, pTarget, obbTree, normalsMesh, inside,first=False):
    """
    This casts a ray from pSource towards pTarget and checks for an intersection with the
    mesh obbTree.  Normal vector from normalsMesh helps determine if the particle being casted is
    inside or outside of the ice.

    Inputs:
        pSource - (3D vector) initial location of photon
        pTarget - (3D unit vector) Final location of photon assuming straight line
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object
        inside - (bool) flag to determine if the particle intial location is within or outside of the ice, helps ensure correct intersection is used.
        first (optional : bool) flag to determine if this is intial point, if so, it does not check for inside.

    Returns:
        intersectionPt - (3D vector) location of intersection
        cellIdIntersection - (int) id of mesh cell where intersection occurs
        normalMeshIntersection - (3D vector) normal vector of intersected cell
        isHit - (bool) True/False flag indicating whether or not an intersection was found
    """
    # Check for intersection with line
    pointsIntersection = vtk.vtkPoints()   # Contains intersection point coordinates
    cellIds = vtk.vtkIdList()              # Contains intersection cell ID

    # Perform intersection test

    code = obbTree.IntersectWithLine(pSource, pTarget, pointsIntersection, cellIds)

    # Total number of intersection points
    numIntersections = pointsIntersection.GetNumberOfPoints()


    if code == 0:
        isHit = False

        # Return empty variables
        intersectionPt = np.array([])
        cellIdIntersection = []
        normalMeshIntersection = []

    else:
        isHit = True

        pointsIntersectionData = pointsIntersection.GetData()

        # Cycle through intersection points until reaching surface that has
        # correct normal direction
        found = False
        idx = 0

        v_i = pts2unitVec(pSource, pTarget).squeeze()
        while found == False:
            # Check if there are any intersection points left
            if idx >= numIntersections:
                isHit = False

                # Return empty variables
                intersectionPt = np.array([])
                cellIdIntersection = []
                normalMeshIntersection = []

                found = True

            else:
                v_n = np.array(normalsMesh.GetTuple(cellIds.GetId(idx)))
                ## Special, check to make sure that the intersection makes sense.
                ## (i.e., the normal direction vector is pointing the right way according to whether or not you are inside the ice.)
                if first == True:## first point,don't check!
                    intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                    cellIdIntersection = cellIds.GetId(idx)
                    normalMeshIntersection = v_n
                    found = True
                    break
                else:
                    if np.dot(v_i, v_n) > 0.0:
                        if inside:
                            intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                            cellIdIntersection = cellIds.GetId(idx)
                            normalMeshIntersection = v_n
                            found = True
                            break
                    else:
                        if not inside:
                            v_n = np.array(normalsMesh.GetTuple(cellIds.GetId(idx)))
                            intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                            cellIdIntersection = cellIds.GetId(idx)
                            normalMeshIntersection = v_n

                            found = True
                            break
            idx += 1

    return intersectionPt, cellIdIntersection, normalMeshIntersection, isHit


def TracktoExt_multi(CRRELPolyData,pSource,vectors):
    """ DEVELOPMENTAL FUNCTION FOR VECTORIZED RAY-CASTING WITH PYVISTA!  STILL IN EARLY DEVELOPMENT
        I HIGHLY RECOMMEND THAT YOU DO NOT USE THIS!!!

        Function attempts to use pyvista to track muliple particles at once in order to run faster.  While this seems to work
        It does not return the same answer as the serial TracktoExt function, and has some known issues with respect to rays "missing"
        intersections.  May be useful in the future!

        This also requires some additional dependencies, and manual modification to the pyvista code in site-packages.

        Inputs:
            CRRELPolyData - (CRREL Poly Data Object) contains required mesh information
            pSource - (3D vector) initial location of photon
            pDir - (3D unit vector) intial direction of photon travel
            raylen - (float: optional) how far to cast ray in search of intersections (mm)

        Returns:
            Information, not yet provided.
        """

    import pyvista as pv

    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."


    pvmesh=pv.wrap(CRRELPolyData.GetPolyData()).triangulate()
    points, ind_ray, ind_tri = pvmesh.multi_ray_trace(pSource, vectors,first_point=True,retry=True)

    pSource=np.array(pSource)[ind_ray,:]

    dists=points-pSource
    dists=np.sqrt( dists[:,0]**2 + dists[:,1]**2 + dists[:,2]**2 )

    return dists,points,ind_tri,ind_ray,pvmesh

def TracktoExt(CRRELPolyData,pSource,pDir,raylen=1000,AirOnly=False):
    import pyvista as pv
    """
    This function sets a particle initial location / direction and looks for the first mesh intersection.
    Returns np.nan if the particle doesn't find an intersection, or if something else goes wrong.

    Inputs:
        CRRELPolyData - (CRREL Poly Data Object) contains required mesh information
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        raylen - (float: optional) how far to cast ray in search of intersections (mm)

    Returns:
        pathLength - (float) distance traveled until intersection
        dot - (float) dot product between the incident vector and normal vector at intersection, can help determine whether particle is inside or outside of ice.

    """
    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."


    inSnow = True ## yes, we are in the space.
    normalsMesh=CRRELPolyData.GetNormalsMesh()
    obbTree=CRRELPolyData.GetObbTree()

    pSource=np.array(pSource)
    pTarget = pSource + pDir * raylen


    intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castFirst(pSource, pTarget,
                                                                                obbTree, normalsMesh)

    if isHit ==True:
        # Incident ray and surface normal vectors
        v_i = pts2unitVec(pSource, pTarget).squeeze()
        v_n = np.array(normalMeshIntersection).squeeze()

        pathLength = ptsDistance(pSource, intersectionPt)
        dot=np.dot(v_i, v_n)


        if pathLength <= CRRELPolyData.Resolution:
            pathLength,dot=np.nan,np.nan
            return pathLength,dot

        if AirOnly == True and dot > 0:
            pathLength,dot=np.nan,np.nan
        # else:
        #     # Set up rendering object
        #     renderer = vtk.vtkRenderer()
        #     renderer.SetBackground(0, 0, 0) # Background color black
        #
        #     # Add actors and render
        #     # Create a mapper and actor for mesh dataset
        #     mapper = vtk.vtkPolyDataMapper()
        #     mapper.SetInputData(CRRELPolyData.GetPolyData())
        #
        #     actor = vtk.vtkActor()
        #     actor.SetMapper(mapper)
        #     actor.GetProperty().SetOpacity(0.8)
        #     renderer.AddActor(actor)
        #
        #
        #     camera = vtk.vtkCamera()
        #
        #     renderer.SetActiveCamera(camera)
        #
        #     # Render
        #     renderWindow = vtk.vtkRenderWindow()
        #     renderWindow.AddRenderer(renderer)
        #     renderWindow.SetSize(800, 800)
        #     renderWindow.Render()
        #
        #     renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        #     renderWindowInteractor.SetRenderWindow(renderWindow)
        #
        #     addLine(renderer, pSource, intersectionPt)
        #     addPoint(renderer,pSource,color=[0.0,1.0,0.0],radius = 0.07)
        #     #addPoint(renderer,intersectionPt,color=[1.0,1.0,0.0],radius = 0.07)
        #
        #     print(pathLength,dot)
        #
        #     renderWindowInteractor.Initialize()
        #     renderWindowInteractor.Start()
        #
        #     sys.exit()

    else:
        pathLength,dot=np.nan,np.nan


    return pathLength,dot


def TracktoAbsStraight(pSource,pTarget,nIce,normalsMesh,obbTree,
        nAir=1.00003,raylen=1000):

    """
    This is a special function that is not generally used, but included as part of a legacy test.
    Basically, this function attempts to estimate F_ice by computing the fractional ice-path along a straight ray
    (i.e., no reflection/refraction along the path).  This was added as a means to try and reduce computational expense, and may be useful
    in the future.  Typically, this function produces LOWER f_ice than the Kaemfer model due to the fact that it eliminates all internal reflections
    that can added to the ice-path length.  It is not recommended that this function is used in practice, but it is here anyway, and can be
    accessed by setting the optional variable ("straight") = True in PhotonTrack.RayTracing_OpticalProperties().


    Inputs:
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        nIce - (float) real part of the refractive index of ice (set from assumed WaveLength)
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        nAir - (float: optional) refractive index of air
        raylen - (float: optional) how far to cast ray in search of intersections (mm)


    Returns:
        TotalIceLength - (float) total distance (mm) traveled within the ice
        TotalLength - (float) total distance (mm) traveled combined ice/air
    """

    # pSource=np.array(pSource)
    # pTarget = pSource + pDir * raylen

    #intersections = np.reshape(np.array(pSource), (1,3))

    TotalIceLength=0
    TotalLength=0
    Hice=[0]
    Hgaps=[0]

    distances,normals, isHit,intersections = castRayAll(pSource, pTarget,obbTree, normalsMesh)

    if isHit ==True:
        TotalLength=np.sum(distances)
        TotalIceLength=np.sum(np.ma.masked_where(normals<=0,distances).compressed())

        Hgaps=list(np.ma.masked_where(normals>0,distances).compressed())
        Hice=list(np.ma.masked_where(normals<=0,distances).compressed())

    return TotalIceLength,TotalLength,Hgaps, Hice,intersections



def TracktoAbsWPhaseF(pSource,pDir,nIce,kIce,normalsMesh,obbTree,
        nAir=1.00003,raylen=1000,polar=0,maxBounce=100,particle=True,MaxTIRbounce=30):

    """
    This function is essentially the Kaempfer model only without absorption.  In essence, this function initializes
    a photon with a location/direction and tracks it within the mesh until it exits the mesh or undergoes more bounces than max bounce.


    Inputs:
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        nIce - (float) real part of the refractive index of ice (set from assumed WaveLength)
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        nAir - (float: optional) refractive index of air
        raylen - (float: optional) how far to cast ray in search of intersections (mm)
        polar - (float: optional / developmental) flag that allows for the ray to be either horizontally or vertically polarized
        maxBounce - (int: optional) Cuts off photon tracking after this number of bounces.  Can significantly reduce computational expense
        particle - (bool: optional) Computes phase function by comparing scattering angles when air/ice phase are different

        MaxTIRbounce (int: optional) Cuts off photon tracking after this number of consecutive total-internal-reflections.
                                     Reduces computational expense caused by TIR loops

    Returns:
        TotalIceLength - (float) total distance (mm) traveled within the ice
        TotalLength - (float) total distance (mm) traveled combined ice/air
        intersections - (array [nBounces, 3]) array containing location of all intersections, can be useful for debugging.
        weights - (array) contains total weights for the computation of the phase function to be applied to COSPHIS
        COSPHIS - (array) contains the scattering angles used to build the scattering phase function
        Fice_Straight (array) contains the fractional ice path along a straight chord through the medium.
                                Used in the computation of the B parameter.
        num_scatter_events (int) - Specific to computing the extinction coefficient.  Counts a scattering event as occuring
                                   on a particle surface, either through reflection off of the surface, or transmission through.

    """

    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice to start.

    pSource=np.array(pSource)
    pTarget = pSource + pDir * raylen
    pTarget1 = pSource + pDir * 1000. ## need a long ray-length for the straight chord used in Fice_straight

    intersections = np.reshape(np.array(pSource), (1,3))

    TotalIceLength=0
    TotalLength=0
    bounce=0
    TIRbounce=0
    first=True
    num_scatter_events=0.0

    weights=[]
    COSPHIS=[]

    # Calculate Fice along a straight chord through the medium
    distances,normals, isHit,intersections = castRayAll(pSource,pTarget,obbTree, normalsMesh)
    TotalLength=np.sum(distances)

    if particle == True:
        incidentParticle=pts2unitVec(pSource, pTarget).squeeze()
        PhaseWeight=1.0 ## Initial phase weight


    if isHit ==True:
        TotalIceLength=np.sum(np.ma.masked_where(normals<=0,distances).compressed())
        Fice_Straight=TotalIceLength/TotalLength
    else:
        Fice_Straight=0.0

    # Begin again and track a realistic photon path as it bounces through the sample
    TotalIceLength=0
    TotalLength=0
    while inSnow:
        if TIRbounce > MaxTIRbounce: ## This takes care of total internal reflection bounce criteria
            inSnow=False
            ## You've done the max number of bounces allowed, leave!
            break
        if bounce > maxBounce: ## This takes care of total internal reflection bounce criteria
            inSnow=False
            ## You've done the max number of bounces allowed, leave!
            break

        intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                                obbTree, normalsMesh,ice,first=first)

        if isHit ==True:
            # Incident ray and surface normal vectors
            v_i = pts2unitVec(pSource, pTarget).squeeze()
            v_n = np.array(normalMeshIntersection).squeeze()
            dist=ptsDistance(pSource, intersectionPt)
            TotalLength+=dist
            # Check to see if incident ray is inside or outside of dense material
            # Assign indices of refraction values
            if first == True:
                first = False

            if np.dot(v_i, v_n) < 0: ## you are in AIR!
                ### IF COMPUTING "particle" SCATTERING -> then THIS IS THE INCIDENT ANGLE! ##
                if particle == True:
                    incidentParticle = v_i ## incidentParticle is the incident unit-vector ray direction on the particle surface!
                    PhaseWeight=1.0 ## New phase weight= 100% energy at particle surface.
                # Assign indices of refraction values
                n1 = nAir
                n2 = nIce
                # Is ray transmitted or reflected?
                v_i_new, reflected,TIRbounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=bounce)
                if particle == True:
                    ## If doing "particle" phase function, add reflected energy to phase function!
                    ## add to reflected weight! ##
                    v_i_ref, v_i_tran, reflect_init,transmiss_init = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    COSPHI=np.dot(incidentParticle,v_i_ref)
                    COSPHIS.append(COSPHI)
                    weights.append(reflect_init*PhaseWeight)
                    PhaseWeight=PhaseWeight*(1. - reflect_init)
                if reflected:
                    ice = False
                    num_scatter_events+=1
                else:
                    ice = True
            else: ## You are in ice!
                # Assign indices of refraction values
                n1 = nIce
                n2 = nAir
                v_n=-v_n
                # If photon is not absorbed, is photon reflected?
                v_i_new, reflected,TIRbounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=bounce)
                if particle == True:
                    v_i_ref, v_i_tran, reflect_init,transmiss_init = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    COSPHI=np.dot(incidentParticle,v_i_tran)
                    COSPHIS.append(COSPHI)
                    weights.append(transmiss_init*PhaseWeight)
                    PhaseWeight = PhaseWeight * (1. - transmiss_init)  #energy left is whatever is relfected!
                if reflected:
                    ice = True
                else:
                    num_scatter_events+=1
                    ice = False
                    if particle == True and PhaseWeight > 0.01:
                        ## Do up-to 20 more bounces for the internal scattering if more than 1% of the particle energy is left?
                        for pdx in range(20):
                            pSource = np.array(intersectionPt) + 1e-3 * v_i_ref
                            pTarget = np.array(intersectionPt) + 4.0 * v_i_ref ##small to avoid computations.
                            dummy,dummy1, GhostNormPt, isHit = castRay(pSource, pTarget,obbTree, normalsMesh,True,first=first)

                            if isHit == False:
                                #print("Hmm, this should not be false, either something went wrong or you have a giant snow grain")
                                #print("Either way, I'm just moving on.")
                                break

                            v_n = -np.array(GhostNormPt).squeeze()
                            v_i_ref, v_i_tran, reflect_init,transmiss_init = Fresnel(nIce, nAir, v_i_ref, v_n,polar=polar)

                            COSPHI=np.dot(incidentParticle,v_i_tran)
                            COSPHIS.append(COSPHI)
                            weights.append(transmiss_init*PhaseWeight)
                            PhaseWeight = PhaseWeight * (1. - transmiss_init)  #energy left is whatever is relfected!
                            if PhaseWeight < 0.01: #break if it's less than 1%
                                break

                    ## IF you are computing phase function as a particle, AND You are starting in ice, the
                    ## scattered angle is THEN the scattered angle transmitting out of the

                TotalIceLength+=dist

                POA=1.-np.exp(-kIce*dist)
                rand=np.random.uniform(0,1)
                if rand <= POA: #PHOTON = DEAD!
                    break

            ## GET PHASE FUNCTION FOR REFLECTION / TRANSMISSION! ##
            if particle == False:
                v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                COSPHI=np.dot(v_i,v_i_ref)
                weights.append(reflect)
                COSPHIS.append(COSPHI)
                COSPHI=np.dot(v_i,v_i_tran)
                COSPHIS.append(COSPHI)
                weights.append(transmiss)

            ## Get Reflected/Transmitted Weights Amount ##
            pSource = np.array(intersectionPt) + 1e-3 * v_i_new
            pTarget = np.array(intersectionPt) + raylen * v_i_new
            # If there are no intersection points, then the particle is gone!

            intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))
            bounce+=1 ## Add to bounce count!
        else:

            #if ice == False:
                ## simple adjustment to account for distance between ice edge and outer boundary?
            #    TotalLength+=np.nanmean(np.ma.masked_where(normals>0,distances).compressed())
            inSnow = False
            break


    return TotalIceLength,TotalLength,intersections,weights,COSPHIS,Fice_Straight,num_scatter_events


def TracktoAbs(pSource,pDir,nIce,normalsMesh,obbTree,
        nAir=1.00003,raylen=1000,polar=0,maxBounce=100,MaxTIRbounce=30):
    """
    This function is essentially the Kaempfer model only without absorption.  In essence, this function initializes
    a photon with a location/direction and tracks it within the mesh until it exits the mesh or undergoes more bounces than max bounce.


    Inputs:
        pSource - (3D vector) initial location of photon
        pDir - (3D unit vector) intial direction of photon travel
        nIce - (float) real part of the refractive index of ice (set from assumed WaveLength)
        normalsMesh - (vtk object) mesh normal vectors -> Passed from CRRELPolyData object
        obbTree - (vtk object) mesh object that contains information used to find intersections within the ray-tracing
        nAir - (float: optional) refractive index of air
        raylen - (float: optional) how far to cast ray in search of intersections (mm)
        polar - (float: optional / developmental) flag that allows for the ray to be either horizontally or vertically polarized
        maxBounce - (int: optional) cuts of photon tracking after this number of bounces.  Can significantly reduce computational expense
        MaxTIRbounce (int: optional) Cuts off photon tracking after this number of consecutive total-internal-reflections.
                                     Reduces computational expense caused by TIR loops

    Returns:
        TotalIceLength - (float) total distance (mm) traveled within the ice
        TotalLength - (float) total distance (mm) traveled combined ice/air
        intersections - (array [nBounces, 3]) array containing location of all intersections, can be useful for debugging.
        Fice_Straight (array) contains the fractional ice path along a straight chord through the medium.
                                Used in the computation of the B parameter.
        first_path_length (float) - length of path to first intersection (mm)
        num_scatter_events (int) - Specific to computing the extinction coefficient.  Counts a scattering event as occuring
                                   on a particle surface, either through reflection off of the surface, or transmission through.
    """

    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice to start.

    pSource=np.array(pSource)
    pTarget = pSource + pDir * raylen
    pTarget1 = pSource + pDir * 1000.  ## need long raylength to compute Fice along a straight chord.
    intersections = np.reshape(np.array(pSource), (1,3))

    TotalIceLength=0
    TotalLength=0
    num_scatter_events=0
    first_path_length = 0
    bounce=0
    TIRbounce=0
    first = True

    distances,normals, isHit,inters = castRayAll(pSource, pTarget1,obbTree, normalsMesh)
    TotalLength=np.sum(distances)

    if isHit ==True:
        TotalLength=np.sum(distances)
        TotalIceLength=np.sum(np.ma.masked_where(normals<=0,distances).compressed())
        Fice_Straight=TotalIceLength/TotalLength
    else:
        Fice_Straight=0.0

    TotalIceLength=0
    TotalLength=0

    while inSnow:
        if TIRbounce > MaxTIRbounce: ## This takes care of total internal reflection bounce criteria
            inSnow=False

            ## You've done the max number of bounces allowed, leave!
            break
        if bounce > maxBounce: ## This takes care of total internal reflection bounce criteria
            inSnow=False
            ## You've done the max number of bounces allowed, leave!
            break

        intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                                obbTree, normalsMesh,ice,first=first)

        if isHit ==True:
            # Incident ray and surface normal vectors
            v_i = pts2unitVec(pSource, pTarget).squeeze()
            v_n = np.array(normalMeshIntersection).squeeze()
            # Check to see if incident ray is inside or outside of dense material
            # Assign indices of refraction values
            if np.dot(v_i, v_n) < 0: ## you are in AIR!
                # Assign indices of refraction values
                n1 = nAir
                n2 = nIce
                # Is ray transmitted or reflected?
                v_i_new, reflected,TIRbounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=TIRbounce)
                if reflected:
                    ice = False
                    num_scatter_events+=1
                else:
                    ice = True

            else: ## You are in ice!
                # Assign indices of refraction values
                n1 = nIce
                n2 = nAir
                v_n=-v_n
                # If photon is not absorbed, is photon reflected?
                v_i_new, reflected,TIRbounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=TIRbounce)
                if reflected:
                    ice = True
                else:
                    ice = False
                    num_scatter_events+=1

                TotalIceLength+=ptsDistance(pSource, intersectionPt)

                if bounce ==1:
                    first_path_length = ptsDistance(pSource, intersectionPt)


            ## Get Reflected/Transmitted Weights Amount ##
            TotalLength+=ptsDistance(pSource, intersectionPt)
            pSource = np.array(intersectionPt) + 1e-3 * v_i_new
            pTarget = np.array(intersectionPt) + raylen * v_i_new
            # If there are no intersection points, then the particle is gone!

            intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))
            bounce+=1 ## Add to bounce count!
        else:
            inSnow = False
            break


    return TotalIceLength,TotalLength,intersections,Fice_Straight,first_path_length,num_scatter_events


def ParticlePhaseFunction(CRRELPolyData,pSource,pTarget,normalsMesh,obbTree,nIce,kIce,units='um',
        nAir=1.00003,raylen=1000,polar=0,nBounce=10,converg=0.999,absorb=True):

    """Particle Phase Function Computes the Phase Function of a specified particle through monte photon tracking
        by comparing the scattering angle of a photon incident upon the particle surface for the direction determined by pSource and pTarget
        and a series of photon tracks exiting the particle.  In this function, rays are tracked and weighted using Snell's and Fresnel's laws until
        the total 99.9% of the energy has been either scattered or absorbed by the particle.  A max-bounce restriction is placed to limit
        particles that get trapped within a TIR loop.

        Inputs:
            CRRELPolyData - A CRRELPolyData Object that represents the 3D particle mesh: Must have an assigned material.
            pSource - initial 3D position vector
            pTarget - initial 3D target vector used to determine initial photon trajectory incident on the particle
            wavelength - wavelength of incident radiation

            units (optional) - units of the WaveLength (default is micrometers), can also be determined from wavelength, if wavelength is a string.
            nAir (optional) - refractive index of air
            raylen (optional) - specified distance to launch photon in, not important, just needs to be long enough to ensure it will intersect the particle
            polar (optional) - IF polar is a complex number, the fresnel equations will be polarized according to the imaginary part of "polar":
                - if the imaginary part is = 1 the parallel polarization (Rs) is returned
                - otherwise the perpendicular polarization (Rp) is returned
                - if polar is NOT a complex number, assumes unpolarized.
            nBounce (optional) - Max number of bounces allowed before photon is killed
            converg (optional) - photon is killed, once this fraction of the total intial energy is accounted for
            absorb (optional) - If true, some photon energy is absorbed within the particle, which can influence the weights of the scattering angles for the later bounces

        Returns:
            weights - array: fraction of incident energy associated with each scattering angle
            COSPHIS - array: the cosine of the scattering angles
            intersections - list: 3D position vectors for each intersection, can be used to show photon trace through the particle
            Dummy - Boolean:
                - If True, then the initial photon trajectory missed the particle and no scattering angles are returned
                - If False, then there are scattering angles
        """


    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."

    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice (or in this example, Glass)

    assert nBounce > 1, "nBouce must be greater than 1!"

    pSource=np.array(pSource)
    pTarget=np.array(pTarget)

    intersections = np.reshape(np.array(pSource), (1,3))

    incidentV= pts2unitVec(pTarget,pSource).squeeze()
    total_path=0
    absorbed=0
    for b in range(nBounce):
        intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                            obbTree, normalsMesh,ice)
        if b == 0 and isHit == False:
            ## MISSES particle, return all -1s.
            return -1,-1,-1,-1,True
            #return [1],[np.dot(incidentV,incidentV)],[],False
        else:
            if isHit ==True:
                ## First ray hits the particle.
                if b == 0:
                    weights=[]
                    COSPHIS=[]

                # Incident ray and surface normal vectors
                v_i = pts2unitVec(pSource, pTarget).squeeze()
                v_n = np.array(normalMeshIntersection).squeeze()

                # Check to see if ray if incident ray is inside or outside of dense material
                # Assign indices of refraction values
                if np.dot(v_i, v_n) < 0: ## you are in AIR!
                    # Assign indices of refraction values
                    n1 = nAir
                    n2 = nIce
                    v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    ice=True

                else: ## You are in ice!
                    # Assign indices of refraction values
                    n1 = nIce
                    n2 = nAir
                    v_n=-v_n
                    v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    ice=True

                pathLength = ptsDistance(pSource, intersectionPt)
                total_path+=pathLength

                if b == 0:
                    Uvect=pts2unitVec(intersectionPt+ 1. * v_i_ref,intersectionPt).squeeze()
                    weights.append(reflect)
                    currentWeight=transmiss
                    v_i_new=v_i_tran
                else:
                    Uvect=pts2unitVec(intersectionPt+ 1. * v_i_tran,intersectionPt).squeeze()
                    if absorb == True:
                        deltaW=(1.-np.exp(-(kIce*pathLength)))*currentWeight
                        currentWeight=currentWeight-deltaW
                        absorbed+=deltaW
                    weights.append(transmiss*currentWeight)
                    currentWeight=currentWeight*reflect
                    v_i_new=v_i_ref

                pSource= np.array(intersectionPt) + 1e-3 * v_i_new
                pTarget = np.array(intersectionPt) + raylen * v_i_new

                COSPHI=np.dot(incidentV,Uvect)
                COSPHIS.append(COSPHI)

                intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))

                if np.sum(weights) >= converg-absorbed:
                    ## You're close enough, exit the loop!
                    remainder=1.-np.sum(weights)-absorbed
                    absorbed+=remainder
                    break

    ScatAlb=1.-absorbed/(np.sum(weights)+absorbed)
    return weights,COSPHIS,intersections,ScatAlb,False



def AdvParticlePhaseFunction(CRRELPolyData,pSource,pTarget,normalsMesh,obbTree,nIce,kIce,units='um',
        nAir=1.00003,raylen=1000,polar=0,nBounce=10,converg=0.999,absorb=True,TrackThresh=0.1,Tdepth=4):

    """Particle Phase Function Computes the Phase Function of a specified particle through monte photon tracking
        by comparing the scattering angle of a photon incident upon the particle surface for the direction determined by pSource and pTarget
        and a series of photon tracks exiting the particle.  In this function, rays are tracked and weighted using Snell's and Fresnel's laws until
        the total 99.9% of the energy has been either scattered or absorbed by the particle.  A max-bounce restriction is placed to limit
        particles that get trapped within a TIR loop.

        Inputs:
            CRRELPolyData - A CRRELPolyData Object that represents the 3D particle mesh: Must have an assigned material.
            pSource - initial 3D position vector
            pTarget - initial 3D target vector used to determine initial photon trajectory incident on the particle
            wavelength - wavelength of incident radiation

            units (optional) - units of the WaveLength (default is micrometers), can also be determined from wavelength, if wavelength is a string.
            nAir (optional) - refractive index of air
            raylen (optional) - specified distance to launch photon in, not important, just needs to be long enough to ensure it will intersect the particle
            polar (optional) - IF polar is a complex number, the fresnel equations will be polarized according to the imaginary part of "polar":
                - if the imaginary part is = 1 the parallel polarization (Rs) is returned
                - otherwise the perpendicular polarization (Rp) is returned
                - if polar is NOT a complex number, assumes unpolarized.
            nBounce (optional) - Max number of bounces allowed before photon is killed
            converg (optional) - photon is killed, once this fraction of the total intial energy is accounted for
            absorb (optional) - If true, some photon energy is absorbed within the particle, which can influence the weights of the scattering angles for the later bounces

        Returns:
            weights - array: fraction of incident energy associated with each scattering angle
            COSPHIS - array: the cosine of the scattering angles
            intersections - list: 3D position vectors for each intersection, can be used to show photon trace through the particle
            Dummy - Boolean:
                - If True, then the initial photon trajectory missed the particle and no scattering angles are returned
                - If False, then there are scattering angles
        """


    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."

    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice (or in this example, Glass)

    assert nBounce > 1, "nBouce must be greater than 1!"

    pSource=np.array(pSource)
    pTarget=np.array(pTarget)

    intersections = np.reshape(np.array(pSource), (1,3))

    incidentV= pts2unitVec(pTarget,pSource).squeeze()
    total_path=0
    absorbed=0
    ExtraPoints=[]
    ExtraDrct=[]
    ExtraWeight=[]
    for b in range(nBounce):
        intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                            obbTree, normalsMesh,ice)
        if b == 0 and isHit == False:
            ## MISSES particle, return all -1s.
            return -1,-1,-1,-1,True
            #return [1],[np.dot(incidentV,incidentV)],[],False
        else:
            if isHit ==True:
                ## First ray hits the particle.
                if b == 0:
                    weights=[]
                    COSPHIS=[]

                # Incident ray and surface normal vectors
                v_i = pts2unitVec(pSource, pTarget).squeeze()
                v_n = np.array(normalMeshIntersection).squeeze()

                # Check to see if ray if incident ray is inside or outside of dense material
                # Assign indices of refraction values
                if np.dot(v_i, v_n) < 0: ## you are in AIR!
                    # Assign indices of refraction values
                    n1 = nAir
                    n2 = nIce
                    v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    ice=True

                else: ## You are in ice!
                    # Assign indices of refraction values
                    n1 = nIce
                    n2 = nAir
                    v_n=-v_n
                    v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                    ice=True

                pathLength = ptsDistance(pSource, intersectionPt)
                total_path+=pathLength

                if b == 0:
                    if reflect > TrackThresh:
                        ExtraPoints.append(intersectionPt)
                        ExtraDrct.append(v_i_ref)
                        ExtraWeight.append(reflect)

                    else:
                        Uvect=pts2unitVec(intersectionPt+ 1. * v_i_ref,intersectionPt).squeeze()
                        weights.append(reflect)
                        COSPHI=np.dot(incidentV,Uvect)
                        COSPHIS.append(COSPHI)

                    currentWeight=transmiss
                    v_i_new=v_i_tran
                else:

                    if absorb == True:
                        deltaW=(1.-np.exp(-(kIce*pathLength)))*currentWeight
                        currentWeight=currentWeight-deltaW
                        absorbed+=deltaW

                    TransWeight=currentWeight*transmiss

                    if TransWeight > TrackThresh:
                        ExtraPoints.append(intersectionPt)
                        ExtraDrct.append(v_i_tran)
                        ExtraWeight.append(TransWeight)
                    else:
                        Uvect=pts2unitVec(intersectionPt+ 1. * v_i_tran,intersectionPt).squeeze()
                        COSPHI=np.dot(incidentV,Uvect)
                        COSPHIS.append(COSPHI)
                        weights.append(transmiss*currentWeight)

                    currentWeight=currentWeight*reflect
                    v_i_new=v_i_ref

                pSource= np.array(intersectionPt) + 1e-3 * v_i_new
                pTarget = np.array(intersectionPt) + raylen * v_i_new

                intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))

                if np.sum(weights) >= converg-absorbed:
                    ## You're close enough, exit the loop!
                    break
    Cdepth=1
    while(Cdepth) <= Tdepth:
        ExtraPointsNew=[]
        ExtraDrctNew=[]
        ExtraWeightNew=[]

        if len(ExtraPoints) == 0:
            Cdepth=Tdepth+1
            break

        for exdx in range(len(ExtraPoints)):
            pSource= np.array(ExtraPoints[exdx]) + 1e-3 * ExtraDrct[exdx]
            pTarget = np.array(ExtraPoints[exdx]) + raylen * ExtraDrct[exdx]

            intersections = np.vstack((intersections, np.reshape(pSource, (1,3))))

            #print(ExtraPoints[exdx],ExtraDrct[exdx],ExtraWeight[exdx])
            ice=False
            for b in range(nBounce):
                intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                                    obbTree, normalsMesh,ice)
                if b == 0 and isHit == False:
                    ## MISSES particle, return all -1s.
                    Uvect=pts2unitVec(ExtraPoints[exdx]+ 1. * ExtraDrct[exdx],ExtraPoints[exdx]).squeeze()
                    weights.append(ExtraWeight[exdx])
                    COSPHI=np.dot(incidentV,Uvect)
                    COSPHIS.append(COSPHI)
                    #intersections = np.vstack((intersections, np.reshape(ExtraPoints[exdx]+ 1. * ExtraDrct[exdx], (1,3))))
                    break
                else:
                    if isHit ==True:
                        # Incident ray and surface normal vectors
                        v_i = pts2unitVec(pSource, pTarget).squeeze()
                        v_n = np.array(normalMeshIntersection).squeeze()

                        # Check to see if ray if incident ray is inside or outside of dense material
                        # Assign indices of refraction values
                        if np.dot(v_i, v_n) >= 0 and b == 0:
                            ## THIS IS A FALSE ALARM!
                            Uvect=pts2unitVec(ExtraPoints[exdx]+ 1. * ExtraDrct[exdx],ExtraPoints[exdx]).squeeze()
                            weights.append(ExtraWeight[exdx])
                            COSPHI=np.dot(incidentV,Uvect)
                            COSPHIS.append(COSPHI)
                            #intersections = np.vstack((intersections, np.reshape(ExtraPoints[exdx]+ 1. * ExtraDrct[exdx], (1,3))))
                            break

                        if np.dot(v_i, v_n) < 0: ## you are in AIR!
                            # Assign indices of refraction values
                            n1 = nAir
                            n2 = nIce
                            v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                            ice=True

                        else: ## You are in ice!
                            # Assign indices of refraction values
                            n1 = nIce
                            n2 = nAir
                            v_n=-v_n
                            v_i_ref, v_i_tran, reflect,transmiss = Fresnel(n1, n2, v_i, v_n,polar=polar)
                            ice=True

                        pathLength = ptsDistance(pSource, intersectionPt)
                        total_path+=pathLength

                        if b == 0:

                            if reflect*ExtraWeight[exdx] > TrackThresh and Cdepth != Tdepth:
                                ExtraPointsNew.append(intersectionPt)
                                ExtraDrctNew.append(v_i_ref)
                                ExtraWeightNew.append(reflect*ExtraWeight[exdx])

                            else:
                                Uvect=pts2unitVec(intersectionPt+ 1. * v_i_ref,intersectionPt).squeeze()
                                weights.append(reflect*ExtraWeight[exdx])
                                COSPHI=np.dot(incidentV,Uvect)
                                COSPHIS.append(COSPHI)

                            currentWeight=transmiss*ExtraWeight[exdx]
                            v_i_new=v_i_tran
                        else:

                            if absorb == True:
                                deltaW=(1.-np.exp(-(kIce*pathLength)))*currentWeight
                                currentWeight=currentWeight-deltaW
                                absorbed+=deltaW

                            TransWeight=currentWeight*transmiss

                            if TransWeight > TrackThresh and Cdepth != Tdepth:
                                ExtraPointsNew.append(intersectionPt)
                                ExtraDrctNew.append(v_i_tran)
                                ExtraWeightNew.append(TransWeight)
                            else:
                                Uvect=pts2unitVec(intersectionPt+ 1. * v_i_tran,intersectionPt).squeeze()
                                COSPHI=np.dot(incidentV,Uvect)
                                COSPHIS.append(COSPHI)
                                weights.append(transmiss*currentWeight)

                            currentWeight=currentWeight*reflect
                            v_i_new=v_i_ref

                        pSource= np.array(intersectionPt) + 1e-3 * v_i_new
                        pTarget = np.array(intersectionPt) + raylen * v_i_new

                        intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))

                        if 1-currentWeight >= converg:
                            ## You're close enough, exit the loop!
                            break

        ExtraPoints=ExtraPointsNew[:]
        ExtraDrct=ExtraDrctNew[:]
        ExtraWeight=ExtraWeightNew[:]
        Cdepth+=1

        if np.sum(weights) >= converg-absorbed:
            ## You're close enough, exit the loop!
            break

#    print(absorbed,np.sum(weights),absorbed/np.sum(weights),1.-absorbed/np.sum(weights))
    ScatAlb=1.-absorbed/(np.sum(weights)+absorbed)
    return weights,COSPHIS,intersections,ScatAlb,False
