import imageio
import matplotlib.pyplot as plt
import os

def cfront(nsga2DataX, nsga2DataY, moeadDataX, moeadDataY, filename='fronts.gif'):
  nsga2FrontsX, nsga2FrontsY, moeadFrontsX, moeadFrontsY,  = [], [], [], []

  for nsga2FrontX, nsga2FrontY in zip(nsga2DataX.split(';'), nsga2DataY.split(';')):
    nsga2FrontsX.append(list(map(float, nsga2FrontX.split(','))))
    nsga2FrontsY.append(list(map(float, nsga2FrontY.split(','))))

    for moeadFrontX, moeadFrontY in zip(moeadDataX.split(';'), moeadDataY.split(';')):
      moeadFrontsX.append(list(map(float, moeadFrontX.split(','))))
      moeadFrontsY.append(list(map(float, moeadFrontY.split(','))))

    iterations = len(nsga2FrontsX)
    count = 0
    
    with imageio.get_writer(filename, mode='I', fps=2) as writer:
      for i in range(iterations):
        _ , axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 8))
        ## The first axes is for NSGA-II
        axs[0].scatter(nsga2FrontsX[i], nsga2FrontsY[i], 50)
        axs[0].title.set_text("NSGA-II")
        axs[0].set_xlabel("Volatility")
        axs[0].set_ylabel("Returns")

        ## The second axes is for MOEAD
        axs[1].scatter(moeadFrontsX[i], moeadFrontsY[i], 50)
        axs[1].title.set_text("MOEA/D-DE")
        axs[1].set_xlabel("Volatility")
        axs[1].set_ylabel("Returns")
        plt.suptitle('The Evolution Process via Tracking Pareto Front',fontsize=20)

        plt.savefig('c-' + str(count) + '.png')
        plt.close()

        image = imageio.imread('c-' + str(count) + '.png')
        writer.append_data(image)
        os.remove('c-' + str(count) + '.png')
        count += 1

