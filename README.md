# Author
Gabriel Sanhueza S.

# Instructions

You need Qt5 and QMake to make it work. Just follow the build instructions.

## Build instructions

* mkdir build
* cd build
* qmake-qt5 ..
* make
* ./qshaderviewer

## How to use this application

Application instructions are in *Help > Tutorial*.
If your attribute and/or uniform variables don't share the names of the ones in this applications, you can modify them in the "generateGLProgram()" and "paintGL()" methods of `oglwidget.cpp`.

