import matplotlib.pyplot as plt
import tb_gosl as gosl


def LearningCurves(y_cv, y_training):
    plt.plot(y_cv,'--ro',label='cv-error')
    plt.plot(y_training,'--bo',label='training-error')
    plt.ylabel('error')
    plt.xlabel('number of examples')
    gosl.FigGrid()
    plt.legend(loc=1)
    plt.show()    