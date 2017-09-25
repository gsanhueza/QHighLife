#ifndef _QHIGHLIFE_H_
#define _QHIGHLIFE_H_

#include <QDesktopWidget>
#include <QMainWindow>
#include "about.h"
#include "tutorial.h"

/**
* @brief Namespace used by qhighlife.ui
*
*/
namespace Ui
{
    class QHighLife;
}

/**
* @brief QHighLife class. Contains the whole window, menu bar, inner widget and status bar.
*
*/
class QHighLife : public QMainWindow
{
    Q_OBJECT;

public:
    /**
    * @brief QHighLife class constructor.
    *
    * @param parent p_parent: Parent of the class. Used by Qt.
    */
    explicit QHighLife(QWidget *parent = nullptr);

    /**
    * @brief QHighLife class destructor.
    *
    */
    ~QHighLife();

public slots:

    /**
     * @brief Receiver of a Qt signal when the Help -> Tutorial action is clicked in the window.
     *
     */
    void loadTutorialClicked();

    /**
     * @brief Receiver of a Qt signal when the Help -> About action is clicked in the window.
     *
     */
    void loadAboutClicked();

private:
    Ui::QHighLife *ui;
    Tutorial *m_tutorial;
    About *m_about;
};

#endif
