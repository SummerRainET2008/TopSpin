#coding: utf8
#author: Tian Xia 

from palframe import *
import matplotlib.pyplot as pylab

class FigureDrawer:
  def __init__(self, title: str, xlabel: str, ylabel: str):
    pylab.clf()
    self._fig, self._ax = pylab.subplots()

    pylab.title(title)
    pylab.grid(linestyle='--', linewidth=0.5)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

  def add_data(self, y_data: list, label: str, x_data: list=None):
    if x_data is None:
      self._ax.plot(y_data, label=label)
    else:
      self._ax.plot(x_data, y_data, label=label)

  def set_xticks(self, xticks):
    self._ax.set_xticks(xticks)

  def set_yticks(self, yticks):
    self._ax.set_yticks(yticks)

  def display(self):
    pylab.legend()
    pylab.show()

  def save_to_figure(self, out_figure: str):
    pylab.legend()
    pylab.savefig(out_figure, bbox_inches="tight")

