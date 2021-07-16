import vtk
import numpy as np
import sys
sys.path.append("/Users/rdcrltwl/Desktop/SnowOpticsProject/NewRTM")
from Shapes import DrawShapes

"""Contains stand alone helper functions to facilitate the photon tracking RT code and interaction with rays and 3D mesh elements."""

def HeyNey(PHI,g=0.847):
    """Henyey-Greenstein Phase function approximation"""
    COSPHI=np.cos(PHI)
    P=(1.-g**2.)/((1.+g**2.-2.*g*COSPHI)**(3./2.))
    return P

def PlankFunction(wavelen,T=5778.):
    """Plank Function
        Wavelength must be in meters"""

    c1=1.191042E8
    c2=1.4387752E4
    L=c1/(wavelen**5*(np.exp(c2/(wavelen*T))-1))
    return L

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
    This function determines whether the photon is reflected or transmitted

    n1 -- incident medium's index of refraction
    n2 -- transmission medium's index of refraction
    v_i -- incident ray direction vector
    v_n -- surface normal direction vector
    polar -- if polar is a complex number, it will determine the photons polarization from it's imaginary part:
        1 = Horizontal polarization
        otherwise = Vertical polarization


    Returns:
        reflected and transmitted vectors, as well as reflected / transmitted weights.

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

    n1 -- incident medium's index of refraction
    n2 -- transmission medium's index of refraction
    v_i -- incident ray direction vector
    v_n -- surface normal direction vector
    count -- counter that keeps track of the number of transmissions and reflections

    returns "v_i_new" which is the new incident ray direction that results from
    a transmission or reflection
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

        #print("ref",reflectance,cosi,cosT,Rs,Rp,cosThetai)
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
            #print("Reflect",np.max(np.abs(v_i_new)),reflectance,cosThetat,cosThetai)
            #print("reflected.")
        # Transmit photon:
        # ----------------
        else:
            # Calculate transmission vector
            v_i_new = n1 / n2 * v_i + ( n1 / n2 * cosi - cosT ) * v_n
            #print("REFRACT ---- ",np.max(np.abs(v_i_new)),reflectance,cosThetat,cosThetai)
            reflected = False
            #print(n1,n2,np.degrees(np.arccos(cosT)),v_i_new)
            #print(v_i_new)
            #print(np.dot(v_i,v_n))
            #print(v_n)

            if np.sqrt(np.sum([i**2. for i in v_i_new])) > 1.0001:
                print("Transmission BAD UNIT VECTOR!",v_i_new)
    # Total internal reflectance condition:
    # -------------------------------------
    else:
        # Calculate reflection vector
        TIR+=1
        v_i_new = v_i + 2 * cosThetai * v_n

        #print("Reflect",np.max(np.abs(v_i_new)))
        reflected = True

        if np.sqrt(np.sum([i**2. for i in v_i_new])) > 1.0001:
            print("TIR BAD UNIT VECTOR!",v_i_new)

    return v_i_new, reflected,TIR



def castRay(pSource, pTarget, obbTree, normalsMesh, inside):
    """
    Ray casting section
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

        v_i = np.array(pTarget) - np.array(pSource)

        while not found:

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
                if np.dot(v_i, v_n) > 0.0:

                    if inside:
                        intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                        cellIdIntersection = cellIds.GetId(idx)
                        normalMeshIntersection = v_n

                        found = True

                else:

                    if not inside:
                        intersectionPt = np.array(pointsIntersectionData.GetTuple3(idx))
                        cellIdIntersection = cellIds.GetId(idx)
                        normalMeshIntersection = v_n

                        found = True

            idx += 1

    return intersectionPt, cellIdIntersection, normalMeshIntersection, isHit


def TracktoExt_multi(CRRELPolyData,pSource,vectors):
    """ DEVELOPMENTAL FUNCTION FOR VECTORIZED RAY-CASTING WITH PYVISTA!  STILL IN EARLY DEVELOPMENT
        I HIGHLY RECOMMEND THAT YOU DO NOT USE THIS!!!"""

    import pyvista as pv

    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."


    pvmesh=pv.wrap(CRRELPolyData.GetPolyData()).triangulate()
    points, ind_ray, ind_tri = pvmesh.multi_ray_trace(pSource, vectors,first_point=True,retry=True)

    pSource=np.array(pSource)[ind_ray,:]

    dists=points-pSource
    dists=np.sqrt( dists[:,0]**2 + dists[:,1]**2 + dists[:,2]**2 )

    return dists,points,ind_tri,ind_ray,pvmesh

def TracktoExt(CRRELPolyData,pSource,pTarget,raylen=1000):
    """
    THIS IS A TRIMMED DOWN VERSION OF THE TrackPhoton Function that is designed to determine the Total Extinction probability of a Photon
     OVER A GIVEN DISTANCE!

    # Increment ray path length
    #                totalPathLength += ptsDistance(prevIntersectionPt, intersectionPt)

    # Calculate probability that ray will be absorbed
    #                Pabs = 1 - exp(-k * totalPathLength)
    """
    assert hasattr(CRRELPolyData, 'isCRRELPolyData') == True,"This is not a CRRELPolyData Object, This function can only take CRRELPolyData Objects."


    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice
    normalsMesh=CRRELPolyData.GetNormalsMesh()
    obbTree=CRRELPolyData.GetObbTree()

    pSource=np.array(pSource)
    pTarget=np.array(pTarget)

    intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                            obbTree, normalsMesh,ice)
    if isHit ==True:
        # Incident ray and surface normal vectors
        v_i = pts2unitVec(pSource, pTarget).squeeze()
        v_n = np.array(normalMeshIntersection).squeeze()

        pathLength = ptsDistance(pSource, intersectionPt)
        dot=np.dot(v_i, v_n)
    else:
        pathLength,dot=0.0,0.0

    return pathLength,dot


def TracktoAbs(CRRELPolyData,pSource,pTarget,nIce,normalsMesh,obbTree,units='um',
        nAir=1.00003,raylen=1000,polar=0,straight=True,maxBounce=100):
    """
    THIS IS A TRIMMED DOWN VERSION OF THE TrackPhoton Function that is designed to determine the Total Absorption probability of a Photon
     OVER A GIVEN DISTANCE!

    # Increment ray path length
    #                totalPathLength += ptsDistance(prevIntersectionPt, intersectionPt)

    # Calculate probability that ray will be absorbed
    #                Pabs = 1 - exp(-k * totalPathLength)
    """

    inSnow = True ## yes, we are in the space.
    ice = False #No we are not in Ice (or in this example, Glass)

    pSource=np.array(pSource)
    pTarget=np.array(pTarget)

    intersections = np.reshape(np.array(pSource), (1,3))

    TotalIceLength=0
    TotalLength=0
    bounce=0
    while inSnow:
        if bounce > maxBounce:
            ## You've done the max number of bounces allowed, leave!
            break

        intersectionPt, cellIdIntersection, normalMeshIntersection, isHit = castRay(pSource, pTarget,
                                                                                obbTree, normalsMesh,ice)
        if isHit ==True:
            # Incident ray and surface normal vectors
            v_i = pts2unitVec(pSource, pTarget).squeeze()
            v_n = np.array(normalMeshIntersection).squeeze()
            # Check to see if ray if incident ray is inside or outside of dense material
            # Assign indices of refraction values
            if np.dot(v_i, v_n) < 0: ## you are in AIR!
                # Assign indices of refraction values
                n1 = nAir
                n2 = nIce
                # Is ray transmitted or reflected?
                if straight == False:
                    v_i_new, reflected,bounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=bounce)
                    if reflected:
                        ice = False
                    else:
                        ice = True

                else:
                    ## continue in straight line, so v_i_new = intersection point.
                    v_i_new = v_i
                    ice = True

            else: ## You are in ice!
                # Assign indices of refraction values
                n1 = nIce
                n2 = nAir
                v_n=-v_n
                # If photon is not absorbed, is photon reflected?
                if straight == False:
                    v_i_new, reflected,bounce = isReflected(n1, n2, v_i, v_n,polar=polar,TIR=bounce)
                    if reflected:
                        ice = True
                    else:
                        ice = False
                else:
                    ## continue in straight line, so v_i_new = intersection point.
                    v_i_new = v_i
                    ice = False

                TotalIceLength+=ptsDistance(pSource, intersectionPt)

            ## Get Reflected/Transmitted Weights Amount ##

            TotalLength+=ptsDistance(pSource, intersectionPt)
            pSource = np.array(intersectionPt) + 1e-3 * v_i_new
            pTarget = np.array(intersectionPt) + raylen * v_i_new
            # If there are no intersection points, then the particle is gone!

            intersections = np.vstack((intersections, np.reshape(intersectionPt, (1,3))))
        else:
            inSnow = False
            break

    return TotalIceLength,TotalLength,intersections


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
            return -1,-1,-1,True
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
                    break
#    print("TOTAL PATH LENGTH (mm) = %s | Total absorbed = %s"%(total_path,absorbed))
    return weights,COSPHIS,intersections,False
