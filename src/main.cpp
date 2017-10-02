#include <QApplication>
#include <QSurfaceFormat>
#include "qhighlife.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QHighLife viewer;

    QSurfaceFormat glFormat;
    glFormat.setVersion(3, 3);
    glFormat.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(glFormat);

    viewer.show();

    return app.exec();
}
