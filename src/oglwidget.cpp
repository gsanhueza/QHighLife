#include <iostream>
#include "oglwidget.h"
#include <cmath>

OGLWidget::OGLWidget(QWidget* parent)
    : QOpenGLWidget(parent),
      m_program(0),
      m_xRot(0),
      m_yRot(0),
      m_zRot(0),
      m_xCamPos(0),
      m_yCamPos(0),
      m_zCamPos(-10),
      m_width(0),
      m_height(0),
      m_grid(nullptr),
      m_gridReader(nullptr)
{
}

OGLWidget::~OGLWidget()
{
    cleanup();
}

void OGLWidget::cleanup()
{
    makeCurrent();
    m_vbo.destroy();
    delete m_program;
    m_program = nullptr;
    doneCurrent();
}

void OGLWidget::setupVertexAttribs()
{
    m_vbo.bind();
    QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0); // Vertex
    f->glEnableVertexAttribArray(1); // Alive
    // glVertexAttribPointer(GLuint index​, GLint size​, GLenum type​, GLboolean normalized​, GLsizei stride​, const GLvoid * pointer​);
    // index = Vertex(0) or Alive(1), can be more if needed
    // size = Coordinates(x, y) => 2
    // type = GL_INT, as that's the type of each coordinate
    // normalized = false, as there's no need to normalize here
    // stride = 0, which implies that vertices are side-to-side (VVVAAA)
    // pointer = where is the start of the data (in VVVAAA, 0 = start of vertices and GL_FLOAT * size(vertexArray) = start of alive status)
    f->glVertexAttribPointer(0, 3, GL_INT, GL_FALSE, 0, 0);
    f->glVertexAttribPointer(1, 3, GL_INT, GL_FALSE, 0, reinterpret_cast<void *>(sizeof(int) * m_data.size() / 2));
    m_vbo.release();
}

void OGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    generateGLProgram();
}

void OGLWidget::generateGLProgram()
{
    m_program = new QOpenGLShaderProgram;
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
        "#version 330 core\n"
        "attribute vec3 vertex;\n"
        "attribute vec3 alive;\n"
        "varying vec3 isAlive;\n"
        "uniform mat4 projMatrix;\n"
        "uniform mat4 modelViewMatrix;\n"
        "void main() {\n"
        "   isAlive = alive;"
        "   gl_Position = projMatrix * modelViewMatrix * vec4(vertex, 1.0);\n"
        " }\n"
    );
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
        "#version 330 core\n"
        "varying vec3 isAlive;\n"
        "void main() {\n"
        "   gl_FragColor = vec4(isAlive, 1.0);\n"
        "}\n"
    );
    m_program->bindAttributeLocation("vertex", 0);
    m_program->bindAttributeLocation("alive", 1);
    m_program->link();

    m_program->bind();
    m_modelViewMatrixLoc = m_program->uniformLocation("modelViewMatrix");
    m_projMatrixLoc = m_program->uniformLocation("projMatrix");

    // Create a vertex array object. In OpenGL ES 2.0 and OpenGL 2.x
    // implementations this is optional and support may not be present
    // at all. Nonetheless the below code works in all cases and makes
    // sure there is a VAO when one is needed.
    m_vao.create();
    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

    // Here you can write fixed geometries, if you don't need to load them more than once.

    // Our camera has a initial position.
    m_camera.setToIdentity();
    m_camera.translate(m_xCamPos, m_yCamPos, m_zCamPos);

    m_program->release();
}

void OGLWidget::loadData(GridReader *gridReader)
{
    // Setup our vertex buffer object.
    m_vbo.create();
    m_vbo.bind();

    // Clear old geometry data from vector.
    m_data.clear();

    // Load geometry (vertices) from grid reader
    m_height = gridReader->getDetectedHeight();
    m_width = gridReader->getDetectedWidth();

    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            // Triangle 1
            m_data.append(i);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j + 1);
            m_data.append(0);

            m_data.append(i);
            m_data.append(-j + 1);
            m_data.append(0);

            // Triangle 2
            m_data.append(i);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j + 1);
            m_data.append(0);
        }
    }

    // Load cell status (alive) from grid reader
    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            for (int k = 0; k < 6; k++)                     // Hack that allow us to map this to each triangle
            {
                m_data.append(gridReader->getData().at(j).at(i) == QChar('1') ? 1 : 0);
                m_data.append(0);
                m_data.append(0);
            }
        }
    }

    // Allocate data into VBO
    m_vbo.allocate(m_data.constData(), m_data.count() * sizeof(int));

    // Store the vertex attribute bindings for the program.
    setupVertexAttribs();

    m_xCamPos = m_width / 2;
    m_yCamPos = -m_height / 2 + 1;
    m_zCamPos = -std::max(m_width, m_height) - 2;
    m_camera.setToIdentity();
    m_camera.translate(-m_xCamPos, -m_yCamPos, m_zCamPos);

    update();
}

void OGLWidget::loadData(Grid *grid)
{
    // Setup our vertex buffer object.
    m_vbo.create();
    m_vbo.bind();

    // Clear old geometry data from vector.
    m_data.clear();

    // Load geometry (vertices) from grid
    m_height = grid->getHeight();
    m_width = grid->getWidth();

    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            // Triangle 1
            m_data.append(i);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j + 1);
            m_data.append(0);

            m_data.append(i);
            m_data.append(-j + 1);
            m_data.append(0);

            // Triangle 2
            m_data.append(i);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j);
            m_data.append(0);

            m_data.append(i + 1);
            m_data.append(-j + 1);
            m_data.append(0);
        }
    }

    // Load cell status (alive) from grid
    for (int j = 0; j < m_height; j++)
    {
        for (int i = 0; i < m_width; i++)
        {
            for (int k = 0; k < 6; k++)                     // Hack that allow us to map this to each triangle
            {
                m_data.append(0);
                m_data.append(grid->getAt(i, j) ? 1 : 0);
                m_data.append(0);
            }
        }
    }

    // Allocate data into VBO
    m_vbo.allocate(m_data.constData(), m_data.count() * sizeof(GLint));

    // Store the vertex attribute bindings for the program.
    setupVertexAttribs();

    m_xCamPos = m_width / 2;
    m_yCamPos = -m_height / 2 + 1;
    m_zCamPos = -std::max(m_width, m_height) - 2;
    m_camera.setToIdentity();
    m_camera.translate(-m_xCamPos, -m_yCamPos, m_zCamPos);

    update();
}

void OGLWidget::paintGL()
{
    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    m_world.setToIdentity();

    // Allow rotation of the world
    m_world.rotate(m_xRot / 16.0f, 1, 0, 0);
    m_world.rotate(m_yRot / 16.0f, 0, 1, 0);
    m_world.rotate(m_zRot / 16.0f, 0, 0, 1);

    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

    // Bind data of shaders to program
    m_program->bind();
    m_program->setUniformValue(m_projMatrixLoc, m_proj);
    m_program->setUniformValue(m_modelViewMatrixLoc, m_camera * m_world);

    // Load new data only on geometry or shader change
    if (m_gridReader != nullptr)
    {
        loadData(m_gridReader);
    }

    if (m_grid != nullptr)
    {
        loadData(m_grid);
    }

    // Draw rectangles as 2 triangles
    glDrawArrays(GL_TRIANGLES, 0, m_data.count() / 3);   // Last argument = Number of vertices

    m_program->release();
}

void OGLWidget::resizeGL(int w, int h)
{
    m_proj.setToIdentity();
    m_proj.perspective(45.0f, GLfloat(w) / h, 0.01f, 200.0f);
}

void OGLWidget::receiveGridReader(GridReader *gridReader)
{
    std::cout << "GridReader received" << std::endl;
    m_gridReader = gridReader;
    m_grid = nullptr;
    m_program = nullptr;
    generateGLProgram();
    update();
}

void OGLWidget::receiveGrid(Grid *grid)
{
    std::cout << "Grid received" << std::endl;
    m_grid = grid;
    m_program = nullptr;
    generateGLProgram();
    update();
}

void OGLWidget::keyPressed(QKeyEvent *event)
{
    // Plus and Minus keys move the camera
    // WASDQE move the light
    switch(event->key())
    {
        // Camera movement
        case Qt::Key_Plus:
            m_zCamPos += 1;
            break;
        case Qt::Key_Minus:
            m_zCamPos -= 1;
            break;
        case Qt::Key_Left:
            m_xCamPos -= 1;
            break;
        case Qt::Key_Right:
            m_xCamPos += 1;
            break;
        case Qt::Key_Up:
            m_yCamPos += 1;
            break;
        case Qt::Key_Down:
            m_yCamPos -= 1;
            break;
        // Reset
        case Qt::Key_Space:
            m_xRot = m_yRot = m_zRot = 0;
            m_xCamPos = m_width / 2;
            m_yCamPos = -m_height / 2 + 1;
            m_zCamPos = -std::max(m_width, m_height) - 2;
        default:
            break;
    }
    m_camera.setToIdentity();
    m_camera.translate(-m_xCamPos, -m_yCamPos, m_zCamPos);
    update();
}

void OGLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}

void OGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(m_xRot + 8 * dy);
        setYRotation(m_yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(m_xRot + 8 * dy);
        setZRotation(m_zRot + 8 * dx);
    }
    m_lastPos = event->pos();
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void OGLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        update();
    }
}

void OGLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        update();
    }
}

void OGLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        update();
    }
}
