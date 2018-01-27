TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt
TARGET = "hard_debug"

INCLUDEPATH += "C:\Program Files (x86)\AMD APP SDK\3.0\include"

LIBS += -L'C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86/' -lOpenCL
INCLUDEPATH += 'C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86'
DEPENDPATH += 'C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86'

SOURCES += \
    main.cpp \
    OpenCLWrapper.cpp

DISTFILES += \
    hard_kernel_копия.cl \
    mat_mul_kernel.cl

HEADERS += \
    OpenCLWrapper.h \
    Matx.h \
    OpenCLTask.h \
    OpenCLQueue.h \
    OpenCLUtility.h
