#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QHighLife viewer;

    viewer.show();

    return app.exec();
}
