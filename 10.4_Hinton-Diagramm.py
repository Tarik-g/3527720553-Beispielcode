import numpy as np
import matplotlib.pyplot as plt

def rect(x,y,sx,sy,col): # zeichne gefülltes Rechteck
    xc = np.array([x,x+sx,x+sx,x])
    yc = np.array([y,y,y+sy,y+sy])
    plt.fill(xc, yc, col, edgecolor=col)

def hinton(matrix): # zeichne Hinton-Diagramm
    plt.clf()
    plt.axis('off')
    plt.axis('equal')
    height, width = matrix.shape
    rect(0,0,width,height,'gray')

    for x in range(width):
        for y in range(height):
            w = matrix[y][x]
            sz = np.sqrt(abs(w)/np.abs(matrix).max()/8)
            col = 'white' if w > 0 else 'black'
            rect(x+0.5-sz, y+0.5-sz, 2*sz, 2*sz, col)

if __name__ == '__main__':
    np.random.seed(8216544)
    # Hinton-Diagramm einer Zufallsmatrix
    hinton(np.random.rand(20, 20) - 0.5)
    plt.title('Beispiel Hinton-Diagramm 20x20') 
    plt.show()