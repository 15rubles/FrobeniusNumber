import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform

def DrawSequence(input, lenX, lenY):
    fontSize = 20
    imageSizeY = 0.32
    imageSizeX = 0.18
    fig = plt.figure()
    if imageSizeX*lenX > imageSizeY*lenY:
        size = imageSizeX*lenX
    else:
        size = imageSizeY*lenY
    fig.set_size_inches(size, size)

    fig.text(5, 5, input, color="black", fontsize=20, transform=IdentityTransform(), va='baseline')
    plt.savefig("seq.png")

    plt.show()
