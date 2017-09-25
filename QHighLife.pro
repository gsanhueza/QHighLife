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

HEADERS +=  src/qhighlife.h \
            src/tutorial.h \
            src/about.h \
