#ifndef _OGLWIDGET_H_
#define _OGLWIDGET_H_

#include <QWidget>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QStringRef>
#include <GL/glu.h>
#include <GL/gl.h>

#include "grid.h"
#include "gridreader.h"
#include "model.h"

class OGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT;

public:
    OGLWidget(QWidget *parent = nullptr);
    ~OGLWidget();

public slots:
    void receiveGridReader(GridReader *gridReader);
    void receiveGrid(Grid *grid);
    void keyPressed(QKeyEvent *event);
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

private:
    Model *m_model;

    void setupVertexAttribs();
    void generateGLProgram();
    void loadData(GridReader *gridReader);
    void loadData(Grid *grid);
    void cleanup();

    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo;
    QOpenGLShaderProgram *m_program;
    int m_modelViewMatrixLoc;
    int m_projMatrixLoc;

    int m_lightPosLoc;
    int m_eyePosLoc;
    QMatrix4x4 m_proj;
    QMatrix4x4 m_camera;
    QMatrix4x4 m_world;
    QPoint m_lastPos;

    int m_xRot;
    int m_yRot;
    int m_zRot;

    float m_xCamPos;
    float m_yCamPos;
    float m_zCamPos;

    QVector<int> m_data;
    int m_width;
    int m_height;
    bool m_dataAlreadyLoaded;
};
#endif
