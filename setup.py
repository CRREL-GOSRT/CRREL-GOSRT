from distutils.core import setup

setup(
    name='CRREL-GSORT',
    version='0.5.0',
    author='Ted Letcher',
    author_email='Theodore.W.Letcher@erdc.dren.mil',
    packages=['crrelGOSRT', 'solarposition'],
    scripts=['bin/Test_Optical_Properties.py','bin/Test_Slab_Model.py'],
    url='https://github.com/wxted/CRREL-GOSRT',
    license='LICENSE.txt',
    description='A blended photon-tracking model for Snow radiative transfer in the geometric optics limit.',
    long_description=open('README.txt').read(),
    install_requires=[
        "vtk >= 9.0.0",
        "pyvista >= 0.29.0",
        "pymeshfix >= 0.14.0"
        "pandas >= 1.0"
    ],
)
