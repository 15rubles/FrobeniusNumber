import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
import numpy as np
import math


def MakePlots(inputNumbers):
    list_of_returns =[]
    nums = []
    numsWithOutCheck = []
    # input numbers
    nums = inputNumbers.copy()
    numsWithOutCheck = inputNumbers.copy()
    divisors = []
    addToFrobenius = 0

    def Multiplier(divs):
        multiplier = 1
        for div in divs:
            multiplier *= div
        return multiplier

    # make numbers coprime
    for i in range(3):
        for j in range(3):
            if i != j:
                divider = math.gcd(nums[i], nums[j])
                if divider > 1:
                    nums[i] //= divider
                    nums[j] //= divider
                    addToFrobenius += Multiplier(divisors) * nums[3 - i - j] * (divider - 1)
                    divisors.append(divider)
    list_of_returns.append(nums)
    list_of_returns.append(divisors)

    # sort numbers
    nums.sort(reverse=True)
    numsWithOutCheck.sort(reverse=True)

    # create graph
    G = nx.DiGraph()

    # add edges to graph
    edge_labels = {}
    for i in range(nums[0]):
        G.add_edge(i, (i + nums[1]) % nums[0], weight=nums[1])
        G.add_edge(i, (i + nums[2]) % nums[0], weight=nums[2])
        edge_labels[(i, (i + nums[1]) % nums[0])] = nums[1]
        edge_labels[(i, (i + nums[2]) % nums[0])] = nums[2]

    # array of all path's, weights from any vertex to any vertex
    len_path = dict(nx.all_pairs_dijkstra(G))

    # find path with max sum of weights
    max_length = 0
    list_of_max = []
    for i in range(len(len_path[0][0])):
        if len_path[0][0][i] == max_length:
            list_of_max.append(i)
        if len_path[0][0][i] > max_length:
            list_of_max.clear()
            list_of_max.append(i)
            max_length = len_path[0][0][i]

    # create arrays of edges from vertices path
    longestPath1 = []
    for i in range(len(len_path[0][1][list_of_max[0]]) - 1):
        longestPath1.append((len_path[0][1][list_of_max[0]][i], len_path[0][1][list_of_max[0]][i + 1]))
    longestPath2 = []
    if len(list_of_max) == 2:
        for i in range(len(len_path[0][1][list_of_max[1]]) - 1):
            longestPath2.append((len_path[0][1][list_of_max[1]][i], len_path[0][1][list_of_max[1]][i + 1]))

    # find edges with b and c weights
    bEdges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == nums[1]]
    cEdges = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == nums[2]]

    # graph layout
    pos = nx.circular_layout(sorted(G.nodes(), reverse=True))
    # pos = nx.circular_layout(G)
    # pos = nx.spiral_layout(sorted(G.nodes(), reverse=False))
    # pos = nx.circular_layout(sorted(G.nodes(), reverse=False))
    # pos = nx.spiral_layout(sorted(G.nodes(), reverse=False))

    # const
    node_sizes = 700
    node_color = "orange"
    arrowSize = 10
    edgeAlpha = 0.5
    maxPathAlpha = 0.6
    maxPathColor = "red"
    bEdgesColor = "black"
    cEdgesColor = "blue"
    arrowStyle = "-|>"
    labelsFont = "sans-serif"
    labelsFontSize = 20
    imageSize = nums[0] + 1
    if imageSize > 50:
        imageSize = 50
    # const

    # M = G.number_of_edges()
    frobNumber = 0

    def FrobeniusNumber(number):
        return (number - nums[0]) * Multiplier(divisors) + addToFrobenius

    # draw vertices
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)

    # draw edgers with b weights
    nx.draw_networkx_edges(
        G, pos, node_size=node_sizes, arrowstyle=arrowStyle, arrowsize=arrowSize, edge_color=bEdgesColor,
        alpha=edgeAlpha,
        width=2, edgelist=bEdges, style="dashed"
    )
    # draw edgers with c weights
    edges = nx.draw_networkx_edges(
        G, pos, node_size=node_sizes, arrowstyle=arrowStyle, arrowsize=arrowSize, edge_color=cEdgesColor,
        alpha=edgeAlpha,
        width=2, edgelist=cEdges
    )
    # draw path with max sum of weights
    if len(list_of_max) == 1:
        edges = nx.draw_networkx_edges(
            G, pos, node_size=node_sizes, arrowstyle=arrowStyle, arrowsize=arrowSize, edge_color=maxPathColor,
            alpha=maxPathAlpha,
            width=3, edgelist=longestPath1, connectionstyle="arc3,rad=-0.1"
        )
        frobNumber = FrobeniusNumber(len_path[0][0][list_of_max[0]])
    elif len_path[0][0][list_of_max[0]] > len_path[0][0][list_of_max[1]]:
        edges = nx.draw_networkx_edges(
            G, pos, node_size=node_sizes, arrowstyle=arrowStyle, arrowsize=arrowSize, edge_color=maxPathColor,
            alpha=maxPathAlpha,
            width=3, edgelist=longestPath1, connectionstyle="arc3,rad=-0.1"
        )
        frobNumber = FrobeniusNumber(len_path[0][0][list_of_max[0]])
    else:
        edges = nx.draw_networkx_edges(
            G, pos, node_size=node_sizes, arrowstyle=arrowStyle, arrowsize=arrowSize, edge_color=maxPathColor,
            alpha=maxPathAlpha,
            width=3, edgelist=longestPath2, connectionstyle="arc3,rad=-0.1")
        frobNumber = FrobeniusNumber(len_path[0][0][list_of_max[1]])

    # frobenius number
    list_of_returns.append(frobNumber)
    # draw labels of vertices
    nx.draw_networkx_labels(G, pos, font_size=labelsFontSize, font_family=labelsFont,
                            verticalalignment="center_baseline")
    # nx.draw_networkx_edge_labels(
    # G, pos, edge_labels=edge_labels , font_color='black')

    # create plot
    ax = plt.gca()
    ax.set_axis_off()
    fig = plt.gcf()

    # set size of img
    fig.set_size_inches(imageSize, imageSize)
    # disable padding on plot
    plt.tight_layout()
    # save plot
    plt.savefig("graph.png")

    # plot of all path's

    dots = {}
    list_of_borders = []

    # find all dots and size of plot
    bmax, cmax, cIndex, bIndex = [0, 0], [0, 0], 0, 0
    for i in range(len(len_path[0][0])):
        bSum = 0
        cSum = 0
        for j in range(len(len_path[0][1][i]) - 1):
            if (len_path[0][1][i][j], len_path[0][1][i][j + 1]) in bEdges:
                bSum += 1
            else:
                cSum += 1
        if cSum > cmax[0] or (cSum == cmax[0] and bSum > cmax[1]):
            cmax[0] = cSum
            cmax[1] = bSum
            cIndex = i
        if bSum > bmax[1] or (bSum == bmax[1] and cSum > bmax[0]):
            bmax[0] = cSum
            bmax[1] = bSum
            bIndex = i
        dots[i] = (cSum, bSum)
    if dots[cIndex][1] == dots[bIndex][1]:
        list_of_borders.append(dots[cIndex][0] + 1)
        list_of_borders.append(dots[cIndex][1] + 1)
        list_of_max.clear()
        list_of_max.append(cIndex)
    else:
        list_of_borders.append(dots[cIndex][0] + 1)
        list_of_borders.append(dots[bIndex][1] + 1)
        list_of_max.clear()
        list_of_max.append(cIndex)
        list_of_max.append(bIndex)

    def CreatePlot(isVals):

        # const
        xmin, ymin = 0, 0
        ticks_frequency = 1
        fSize = 50
        plotSize = max(list_of_borders[0], list_of_borders[1]) + 2
        if plotSize > 50:
            plotSize = 50
        lineColor = "black"
        lineStyle = "-"
        lineWidth = 4
        ticksSize = 30
        # const


        # create plot adn set size of img
        fig, ax = plt.subplots(figsize=(plotSize, plotSize))

        def SetTicks(xmax, ymax):
            ax.set(xlim=(xmin, xmax + 0.5), ylim=(ymin, ymax + 0.5), aspect='equal')
            ax.tick_params(axis='both', which='major', labelsize=ticksSize)
            ax.tick_params(axis='both', which='minor', labelsize=ticksSize)
            x_ticks = np.arange(xmin, xmax + 2, ticks_frequency)
            y_ticks = np.arange(ymin, ymax + 2, ticks_frequency)
            ax.set_xticks(x_ticks[x_ticks != 0])
            ax.set_yticks(y_ticks[y_ticks != 0])
            ax.set_xticks(np.arange(0, xmax, step=1))
            ax.set_yticks(np.arange(0, ymax, step=1))
            ax.set_xlabel('c', size=ticksSize, labelpad=-30, x=0.97, y=1)
            ax.set_ylabel('b', size=ticksSize, labelpad=-9, y=0.97, rotation=0)

        def PlotADot(xy, value):
            if isVals:
                if isinstance(value, int):
                    div = len(str(value)) - 2
                    if div <= 0:
                        div = 0.5
                    if len(str(value)) == 1:
                        ax.annotate(value, (xy[0] + 0.32, xy[1] + 0.27), fontsize=fSize / math.pow(1.65, div))
                    else:
                        ax.annotate(value, (xy[0] + 0.07, xy[1] + 0.27), fontsize=fSize / math.pow(1.65, div))
                else:
                    ax.annotate(value, (xy[0] + 0.07, xy[1] + 0.27), fontsize=fSize)
            else:
                if isinstance(value, int):
                    div = len(str(value % nums[0])) - 2
                    if div <= 0:
                        div = 0.5
                    if len(str(value % nums[0])) == 1:
                        ax.annotate(value % nums[0], (xy[0] + 0.32, xy[1] + 0.27), fontsize=fSize / math.pow(1.65, div))
                    else:
                        ax.annotate(value % nums[0], (xy[0] + 0.07, xy[1] + 0.27), fontsize=fSize / math.pow(1.65, div))
                else:
                    ax.annotate(value, (xy[0] + 0.07, xy[1] + 0.27), fontsize=fSize)

        SetTicks(list_of_borders[0], list_of_borders[1])

        # enable grid on plot
        plt.grid()
        # plotting the plot
        for i in range(nums[0]):
            PlotADot(dots[i], len_path[0][0][i])
        if len(list_of_max) == 1:
            ax.plot([0, dots[list_of_max[0]][0] + 1], [dots[list_of_max[0]][1] + 1, dots[list_of_max[0]][1] + 1],
                    c=lineColor,
                    ls=lineStyle,
                    lw=lineWidth)
            ax.plot([dots[list_of_max[0]][0] + 1, dots[list_of_max[0]][0] + 1], [dots[list_of_max[0]][1] + 1, 0],
                    c=lineColor,
                    ls=lineStyle,
                    lw=lineWidth)
            ax.plot([0, 0], [0, dots[list_of_max[0]][1] + 1], c=lineColor, ls=lineStyle, lw=lineWidth)
            ax.plot([0, dots[list_of_max[0]][0] + 1], [0, 0], c=lineColor, ls=lineStyle, lw=lineWidth)
            PlotADot([dots[list_of_max[0]][0] + 1, dots[list_of_max[0]][1] + 1], "C")
        else:
            first = 1
            second = 0
            if dots[list_of_max[0]][1] > dots[list_of_max[1]][1]:
                first = 0
                second = 1

            ax.plot([0, dots[list_of_max[first]][0] + 1],
                    [dots[list_of_max[first]][1] + 1, dots[list_of_max[first]][1] + 1],
                    c=lineColor, ls=lineStyle, lw=lineWidth)
            ax.plot([dots[list_of_max[first]][0] + 1, dots[list_of_max[first]][0] + 1],
                    [dots[list_of_max[first]][1] + 1, dots[list_of_max[second]][1] + 1],
                    c=lineColor, ls=lineStyle, lw=lineWidth)
            ax.plot([dots[list_of_max[first]][0] + 1, dots[list_of_max[second]][0] + 1],
                    [dots[list_of_max[second]][1] + 1, dots[list_of_max[second]][1] + 1],
                    c=lineColor, ls=lineStyle, lw=lineWidth)
            ax.plot([dots[list_of_max[second]][0] + 1, dots[list_of_max[second]][0] + 1],
                    [dots[list_of_max[second]][1] + 1, 0],
                    c=lineColor, ls=lineStyle, lw=lineWidth)

            ax.plot([0, 0], [0, dots[list_of_max[first]][1] + 1], c=lineColor, ls=lineStyle, lw=lineWidth)
            ax.plot([0, dots[list_of_max[second]][0] + 1], [0, 0], c=lineColor, ls=lineStyle, lw=lineWidth)
            PlotADot([dots[list_of_max[first]][0] + 1, dots[list_of_max[first]][1] + 1], "C")
            PlotADot([dots[list_of_max[second]][0] + 1, dots[list_of_max[second]][1] + 1], "E")

        plt.tight_layout()
        # save img
        if isVals:
            plt.savefig("Values.png")
        else:
            plt.savefig("Vertices.png")

    CreatePlot(True)
    CreatePlot(False)

    # all numbers under frobenius

    nonRepresentedNums = set()

    for i in range(3):
        possible_numbers = numsWithOutCheck.copy()
        possible_numbers.remove(numsWithOutCheck[i])
        for j in range(2):
            currentNumber = 0
            toAppend1 = []
            while True:
                currentNumber += possible_numbers[j]
                if currentNumber > frobNumber:
                    break
                toAppend1.append(currentNumber)
            toAppend2 = []
            for k in toAppend1:
                nonRepresentedNums.add(k)
                currentNumber = k
                while True:
                    currentNumber += possible_numbers[1 - j]
                    if currentNumber > frobNumber:
                        break
                    toAppend2.append(currentNumber)
            for k in toAppend2:
                nonRepresentedNums.add(k)

    # all numbers in NR set
    NRset = [*range(1, frobNumber + 1, 1)]
    for n in nonRepresentedNums:
        NRset.remove(n)
    list_of_returns.append(NRset)
    # disable padding on plot


    return list_of_returns

MakePlots([7, 3, 5])
