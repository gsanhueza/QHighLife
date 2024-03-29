QT      +=  widgets

TARGET   =  qhighlife
TEMPLATE =  app

FORMS   +=  qhighlife.ui \
            about.ui \
            tutorial.ui \

SOURCES +=  src/main.cpp \
            src/qhighlife.cpp \
            src/tutorial.cpp \
            src/about.cpp \
            src/grid.cpp \
            src/model.cpp \
            src/cpumodel.cpp \
            src/cudamodel.cpp \
            src/openclmodel.cpp \
            src/gridreader.cpp \
            src/canvas.cpp \

HEADERS +=  src/qhighlife.h \
            src/tutorial.h \
            src/about.h \
            src/grid.h \
            src/model.h \
            src/cpumodel.h \
            src/cudamodel.h \
            src/openclmodel.h \
            src/gridreader.h \
            src/canvas.h \

LIBS += -lcuda -lcudart -lOpenCL

#### CUDA Settings ####

CUDA_SOURCES += src/highlife.cu \

# Path to cuda toolkit install
CUDA_DIR      = /opt/cuda

# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # 64 bits Operating system

# GPU architecture
CUDA_ARCH     = sm_20                # Adjust with your compute capability

# NVCC flags
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -Xcompiler=-fPIC -std=c++11

# Prepare the extra compiler configuration
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 \ #-arch=$$CUDA_ARCH \
                -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
