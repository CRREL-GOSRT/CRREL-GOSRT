import vtk

""" A collection of stand alone functions written specifically to help render objects within the vtk 3D
    rendering environment as part of the crrelGOSRT framework.  There are no examples of how to use these
    functions within the crrelGOSRT framework, but they can potentially be helpful for debugging.

    Typical usage first requires calling the "Render3DMesh" function with an input CRRELPolyData object:

    renderer = Render3DMesh(CRRELPolyData)

    This will define a "renderer" object that can be called by the other functions here to draw additional things like
    points or lines on the renderer.

    Finally, the "ShowRender" function is called to display the output.  For example (assuming you have a CRRELPolyData object defined):

        renderer = Render3DMesh(CRRELPolyData)
        p1=(0.5,0.5,0.5)
        p2=(0.1,0.0,1.2)

        addPoint(renderer,p1)
        addPoint(renderer,p2)
        addLine(renderer,p1,p2)
        ShowRender()

    """

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


def Render3DMesh(CRRELPolyData,bcolor=(0,0,0),opacity=0.95,orientation=True,BoundBox=True,renderer=None):
    ## initialize renderer

    """
        Creates a vtk renderer around a 3D mesh wrapped within a CRRELPolyData object.

        Inputs:
            - CRRELPolyData (CRRELPolyData object)

            - bcolor (optional: tuple vector) background color of renderer
            - opacity (optional :float 0-1) opacity of the mesh.
            - orientation (optional : bool) if "True", will draw a small 3D coordinate system orientation on mesh
            - BoundBox (optional : bool) if "True", will draw lines on the boundaries indicated the size of the mesh
            - renderer (optiona : vtk renderer object) if "None", this will create a new renderer, otherwise it will add the mesh to an existing renderer

        Returns:
            renderer (vtk renderer object)
    """

    if isinstance(renderer, type(None)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*bcolor) # Background color black

    if orientation == True:
        x1=CRRELPolyData.xBounds[0]
        y1=CRRELPolyData.yBounds[0]
        z1=CRRELPolyData.zBounds[1]
        x2,y2,z2=x1+1,y1+1,z1+1

        addLine(renderer, [x1,y1,z1], [x2,y1,z1], color=[1.0, 0.0, 0.0], lineWidth=4.0)
        addLine(renderer, [x1,y1,z1], [x1,y2,z1], color=[0.0, 1.0, 0.0], lineWidth=4.0)
        addLine(renderer, [x1,y1,z1], [x1,y1,z2], color=[0.0, 0.0,1.0], lineWidth=4.0)


    if BoundBox == True:
        ## add more lines. -- >x!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        ## add more lines. -- >y!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        ## add more lines. -- >z!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)

    # Add actors and render
    # Create a mapper and actor for mesh dataset
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(CRRELPolyData.GetPolyData())
    #normalsMesh=shell.compute_normals()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)

    return renderer


def ShowRender(renderer,camera=None,size=(800,800)):
    """
        Displays the renderer object on the screen.

        Inputs:
            - renderer (vtk renderer object)

            - camera (optional: vtk camera object) if "None", this will define a default camera, otherwise it will use a camera passed by the user.
            - size (optional: 2D tuple) determines the size (in pixels) of the display window.
        Returns:
            None
    """
    if camera == None:
        camera = vtk.vtkCamera()
        camera.SetPosition(0,-10,10)
        camera.SetFocalPoint(0,0,0)
        camera.SetViewUp(0,0,0)

    #    camera.ParallelProjectionOn()

    renderer.SetActiveCamera(camera)

    # Render
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(*size)
    renderWindow.Render()

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()
