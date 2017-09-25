#include <QApplication>
#include <QSurfaceFormat>
#include "qshaderviewer.h"
#include "model.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Model model;
    QShaderViewer viewer(model);

    QSurfaceFormat glFormat;
    glFormat.setVersion(3, 3);
    glFormat.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(glFormat);

    viewer.show();

    return app.exec();
}
