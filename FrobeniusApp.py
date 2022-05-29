import dearpygui.dearpygui as dpg
# from plotsAndNumbers import MakePlots
# from frobeniusFromSequence import Sequence
# from stringToImg import DrawSequence
from PIL import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib.transforms import IdentityTransform
import matplotlib.pyplot as plt
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
    plt.clf()


def Sequence(input_numbers):

    nums = input_numbers.copy()

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
    print("divided nums", nums)
    # sort numbers
    nums.sort(reverse=True)
    input_numbers.sort(reverse=True)
    s = [nums[0]]
    # sort numbers
    nums.sort(reverse=True)
    # find S[0]
    s0 = 1
    while (s0 * nums[1]) % nums[0] != nums[2]:
        s0 += 1
    s.append(s0)
    q = []
    i = 0
    while s[-1] != 0:
        i += 1
        next_q = 1
        while s[i]*next_q < s[i-1]:
            next_q += 1
        q.append(next_q)
        s.append(next_q*s[i] - s[i-1])
    print("s", s)
    print("q", q)
    p = [0, 1]
    i = 2
    for q_el in q:
        p.append(q_el*p[i-1]-p[i-2])
        i += 1
    print("p", p)
    s_divided_by_q = []
    s_divided_by_q_print = []
    for i in range(len(s)):
        if p[i] == 0:
            s_divided_by_q.append(float('inf'))
        else:
            s_divided_by_q.append(s[i]/p[i])
            s_divided_by_q_print.append(str(s[i])+"/"+str(p[i]))
    s_divided_by_q_print.reverse()
    print(s_divided_by_q)
    print("Print", s_divided_by_q_print)
    c_divided_by_b = nums[2] / nums[1]
    u = 0
    while not (s_divided_by_q[u] > c_divided_by_b >= s_divided_by_q[u + 1]):
        u += 1
    print("u", u)

    frobeniusNumber = -nums[0] + nums[1]*(s[u]-1)+nums[2]*(p[u+1]-1)-min(nums[1]*s[u+1], nums[2]*p[u])
    print("Frobenius number:", frobeniusNumber)
    frobeniusNumber_with_input_nums = frobeniusNumber * Multiplier(divisors) + addToFrobenius
    print("Frobenius number with input nums:", frobeniusNumber_with_input_nums)
    output = []
    SUB = str.maketrans("0123456789-i+mu", "₀₁₂₃₄₅₆₇₈₉₋ᵢ₊ₘᵤ")
    output.append("Введенные числа: a="+str(input_numbers[0])+", b="+str(input_numbers[1])+", c="+str(input_numbers[2]))
    output.append("Упрощенные числа: a="+str(nums[0])+", b="+str(nums[1])+", c="+str(nums[2])+" (по формуле Джонсона)")
    output.append("")
    output.append("Формулы:")
    output.append("bS₀ ≡ c (mod a)")
    output.append("S-1".translate(SUB)+"= a")
    output.append("Si-2= qiSi-1".translate(SUB)+" - "+"Si".translate(SUB))
    output.append("P-1= ".translate(SUB)+"0"+"  P0= ".translate(SUB)+"1")
    output.append("Pi= qiPi-1".translate(SUB)+" - "+"Pi-2".translate(SUB))
    output.append("")
    output.append("Последовательности {Sᵢ}, {Pᵢ}, {qᵢ}:")
    for i in range(len(s)):
        convert = str(i-1)
        add = ''
        add += "S"+convert.translate(SUB)+"= "+str(s[i])+"  "
        add += "P"+convert.translate(SUB)+"= "+str(p[i])+"  "
        if i < len(q):
            add += "q"+str(i+1).translate(SUB)+"= "+str(q[i])
        output.append(add)
    output.append("Неравенство:")
    output.append("0=Sm+1/Pm+1<Sm/Pm<...<S0/P0<S-1/P-1=∞".translate(SUB))
    s_divided_by_q_print = '<'.join(map(str, s_divided_by_q_print))
    output.append("0="+s_divided_by_q_print+"<"+str(s[0])+"/"+str(p[0])+"=∞")
    output.append("c/b= "+ str(nums[2])+"/"+str(nums[1]))
    output.append("u= "+str(u))
    output.append("")
    output.append("Число Фробениуса для упрощенных чисел:")
    output.append("g(a,b,c)=-a+(Sᵤ-1)+c(P"+"u+1".translate(SUB)+"-1)-min{bS"+"u+1".translate(SUB)+", cPᵤ}")
    output.append("g("+str(nums[0])+","+str(nums[1])+","+str(nums[2])+")=-"+str(nums[0])+"+"+str(nums[1])+"*("+str(s[u])+"-1)"+"+"+str(nums[2])+"*("+str(p[u+1])+"-1)"+"-"+"min{"+str(nums[1])+"*"+str(s[u+1])+", "+str(nums[2])+"*"+str(p[u])+"}="+str(frobeniusNumber))
    output.append("Число Фробениуса для введенных чисел: "+str(frobeniusNumber_with_input_nums))
    return output


size = 2000, 2000
infileGraph = "graph.png"
infileValues = "Values.png"
outfileGraph = "graph_resize.png"
outfileValues = "Values_resize.png"
infileVertices = "Vertices.png"
outfileVertices = "Vertices_resize.png"
infileSeq = "seq.png"
outfileSeq = "seq_resize.png"


def ResizeImage(infile, outfile):
    im = Image.open(infile)
    if GetSize(infile)[0] < size[0]:
        imb = im.resize(size)
        imb.save(outfile, "PNG")
        im.close()
    else:
        im.thumbnail(size, Image.Resampling.LANCZOS)
        im.save(outfile, "PNG")
        im.close()


def GetSize(file):
    width, height = Image.open(file).size
    return [width,height]


dpg.create_context()

with dpg.font_registry():
    with dpg.font("NotoSans.ttf", 20) as font1:
        # add the default font range
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)

plotVars = ["None", "None", "None", "None"]
seq = '\u208a'

enterLength = 80


def Plots():
    global plotVars
    global seq

    seq = Sequence([dpg.get_value("a"), dpg.get_value("b"), dpg.get_value("c")])
    lenY = len(seq)
    lenX = 0
    for i in seq:
        if len(i) > lenX:
            lenX = len(i)

    seq = '\n'.join(map(str, seq))
    plotVars = MakePlots([dpg.get_value("a"), dpg.get_value("b"), dpg.get_value("c")])
    plotVars[3] = ', '.join(map(str, plotVars[3]))
    i = enterLength
    while i < len(plotVars[3]):
        while plotVars[3][i] != " ":
            i -= 1
        plotVars[3] = plotVars[3][:i+1] + '\n' + plotVars[3][i+1:]
        i += enterLength
    plotVars[1] = ', '.join(map(str, plotVars[1]))
    plotVars[0] = ', '.join(map(str, plotVars[0]))
    print(seq)
    DrawSequence(seq, lenX, lenY+1)
    update_dynamic_textures()
    dpg.set_value("nums", plotVars[0])
    dpg.set_value("divisors", plotVars[1])
    dpg.set_value("frobNum", plotVars[2])
    dpg.set_value("NR set", plotVars[3])


ResizeImage(infileGraph, outfileGraph)
width, height, channels, data = dpg.load_image(outfileGraph)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width, height, data, tag="graph")

ResizeImage(infileValues, outfileValues)
width, height, channels, data = dpg.load_image(outfileValues)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width, height, data, tag="val")

ResizeImage(infileVertices, outfileVertices)
width, height, channels, data = dpg.load_image(outfileVertices)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width, height, data, tag="ver")

ResizeImage(infileSeq, outfileSeq)
width, height, channels, data = dpg.load_image(outfileSeq)
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width, height, data, tag="seq")


def update_dynamic_textures():
    ResizeImage(infileGraph, outfileGraph)
    width, height, channels, data = dpg.load_image(outfileGraph)
    dpg.set_value("graph", data)
    ResizeImage(infileValues, outfileValues)
    width, height, channels, data = dpg.load_image(outfileValues)
    dpg.set_value("val", data)
    ResizeImage(infileVertices, outfileVertices)
    width, height, channels, data = dpg.load_image(outfileVertices)
    dpg.set_value("ver", data)
    ResizeImage(infileSeq, outfileSeq)
    width, height, channels, data = dpg.load_image(outfileSeq)
    dpg.set_value("seq", data)


with dpg.window(tag="Primary Window"):
    with dpg.group(horizontal=True):
        with dpg.plot(label="Граф", height=500, width=500):
            dpg.add_plot_axis(dpg.mvXAxis)
            with dpg.plot_axis(dpg.mvYAxis):
                dpg.add_image_series("graph", [0, 0], [100, 100])
        # dpg.add_image("graph")
        dpg.add_spacer()
        with dpg.plot(label="Сумма пути до вершины", height=500, width=500):
            dpg.add_plot_axis(dpg.mvXAxis)
            with dpg.plot_axis(dpg.mvYAxis):
                dpg.add_image_series("val", [0, 0], [100, 100])
        dpg.add_spacer()
        with dpg.plot(label="Путь до вершины", height=500, width=500):
            dpg.add_plot_axis(dpg.mvXAxis)
            with dpg.plot_axis(dpg.mvYAxis):
                dpg.add_image_series("ver", [0, 0], [100, 100])
    dpg.bind_font(font1)
    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            dpg.add_text("Введите 3 числа:")
            dpg.add_input_int(label="a", default_value=7, min_value=1, width=200, tag="a")
            dpg.add_input_int(label="b", default_value=3, min_value=1, width=200, tag="b")
            dpg.add_input_int(label="c", default_value=5, min_value=1, width=200, tag="c")
            dpg.add_button(label="Посчитать", callback=Plots)
            dpg.add_input_text(label="Упрощенные числа", default_value=plotVars[0], width=500, height=100, tag="nums", readonly=True)
            dpg.add_input_text(label="Делители", default_value=plotVars[1], width=500, height=100, tag="divisors", readonly=True)
            dpg.add_input_text(label="Число Фробениуса", default_value=plotVars[2], width=500, height=100, tag="frobNum", readonly=True)
            dpg.add_input_text(label="NR множество", default_value=plotVars[3], width=500, height=200, multiline=True, tag="NR set", readonly=True)
        dpg.add_spacer()
        with dpg.plot(label="Число Фробениуса через цепные дроби", height=500, width=500):
            dpg.add_plot_axis(dpg.mvXAxis)
            with dpg.plot_axis(dpg.mvYAxis):
                dpg.add_image_series("seq", [0, 0], [100, 100])


dpg.create_viewport(title='Frobenius calculator', width=1570, height=900, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()

dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
