cv-gpu-py
==================

Python wrapper for OpenCV C++ gpu::PyrLKOpticalFlow and gpu::OpticalFlowDual_TVL1_GPU . Based on [pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter).

Compatibility
-----------------
This code is tested on OpenCV 2.4.13 with CUDA and python 2.7.

Compiling & Trying Out Sample Code
----------------------
1. Install CMake and/or CMake-gui (http://www.cmake.org/download/, ```sudo apt-get install cmake cmake-gui``` on Ubuntu/Debian)
2. Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
3. Build (run ```make``` on *nix systems with gcc/eclipse CDT generator from within the build folder)
4. On *nix systems, ```make install``` run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version).
5. Run python interpreter of your choice, issue 
  1. import pbcvt; import numpy as np
  2. a = np.array([[1.,2.],[3.,4.]]); b = np.array([[2.,2.],[4.,4.]])
  3. pbcvt.dot(a,b)
  4. pbcvt.dot2(a,b)
