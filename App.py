import dearpygui.dearpygui as dpg
from plotsAndNumbers import MakePlots
from frobeniusFromSequence import Sequence
from stringToImg import DrawSequence
from PIL import Image
import pickle
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
