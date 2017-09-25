#include <QApplication>
#include "qhighlife.h"

extern "C"
void run();

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QHighLife viewer;

    viewer.show();

    return app.exec();
}
