from collections import OrderedDict, namedtuple
from itertools import product
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
import numpy as np
from operator import methodcaller
from os.path import exists
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from . import data

mpl.use('Qt5Agg')


EARTH_RADIUS = 6.371e6

Projection = namedtuple('Projection', ['key', 'name', 'method'])
PROJECTIONS = OrderedDict([
    ('aea', Projection('aea', 'Albers Equal Area', 'width')),
    ('aeqd', Projection('aeqd', 'Azimuthal Equidistant', 'width')),
    ('cass', Projection('cass', 'Cassini-Soldner', 'width')),
    # ('cea', Projection('cea', 'Cylindrical Equal Area', 'degrees')),
    # ('cyl', Projection('cyl', 'Cylindrical Equidistant', 'degrees')),
    # ('eck4', Projection('eck4', 'Eckert IV', 'width')),
    ('eqdc', Projection('eqdc', 'Equidistant Conic', 'width')),
    # ('gall', Projection('gall', 'Gall Stereographic Cylindrical', 'degrees')),
    ('gnom', Projection('gnom', 'Gnomonic', 'width')),
    # ('hammer', Projection('hammer', 'Hammer', 'width')),
    # ('kav7', Projection('kav7', 'Kavrayskiy VII', 'width')),
    ('laea', Projection('laea', 'Lambert Azimuthal Equal Area', 'width')),
    ('lcc', Projection('lcc', 'Lambert Conformal', 'width')),
    # ('mbtfpq', Projection('mbtfpq', 'McBryde-Thomas Flat-Polar Quartic', 'width')),
    # ('merc', Projection('merc', 'Mercator', 'degrees')),
    # ('mill', Projection('mill', 'Miller Cylindrical', 'degrees')),
    # ('moll', Projection('moll', 'Mollweide', 'width')),
    # ('nsper', Projection('nsper', 'Near-Sided Perspective', 'degrees')),
    # ('npaeqd', Projection('npaeqd', 'North-Polar Azimuthal Equidistant', 'width')),
    # ('nplaea', Projection('nplaea', 'North-Polar Lambert Azimuthal', 'width')),
    # ('npstere', Projection('npstere', 'North-Polar Stereographic', 'width')),
    # ('omerc', Projection('omerc', 'Oblique Mercator', 'width')),
    # ('ortho', Projection('ortho', 'Orthographic', 'width')),
    ('poly', Projection('poly', 'Polyconic', 'width')),
    # ('robin', Projection('robin', 'Robinson', 'width')),
    # ('rotpole', Projection('rotpole', 'Rotated Pole', 'width')),
    # ('sinu', Projection('sinu', 'Sinusoidal', 'width')),
    # ('spaeqd', Projection('spaeqd', 'South-Polar Azimuthal Equidistant', 'width')),
    # ('splaea', Projection('splaea', 'South-Polar Lambert Azimuthal', 'width')),
    # ('spstere', Projection('spstere', 'South-Polar Stereographic', 'width')),
    ('stere', Projection('stere', 'Stereographic', 'width')),
    # ('tmerc', Projection('tmerc', 'Transverse Mercator', 'width')),
    # ('vandg', Projection('vandg', 'van der Grinten', 'degrees')),
])

CONTOURS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]


class BlockDrawer:

    def __init__(self, block, draw, selected):
        self.block = block
        self.selected = selected
        self.draw_field = draw
        self.line = None
        self.contour = None
        self.barbs = None

    def draw(self, m, clear=False):
        coords = [m(*p) for p in self.block.pts]
        coords.append(coords[0])

        cx = [c[0] for c in coords]
        cy = [c[1] for c in coords]
        color = 'white'
        linestyle = 'solid' if self.selected else 'dotted'
        linewidth = 2 if self.selected else 1
        zorder = 11 if self.selected else 10

        if clear or not self.line:
            self.line = m.plot(cx, cy, color=color,
                               linestyle=linestyle,
                               linewidth=linewidth,
                               zorder=zorder)[0]
        else:
            self.line.set_data(cx, cy)
            self.line.set_color(color)
            self.line.set_linestyle(linestyle)
            self.line.set_linewidth(linewidth)
            self.line.zorder = zorder

        if self.draw_field:
            self.block.compute()
            if clear or not self.contour:
                self.contour = m.contourf(
                    self.block.lons, self.block.lats, self.block.data,
                    latlon=True, zorder=5, alpha=0.6,
                    levels=CONTOURS, vmin=0, vmax=24
                )
            if clear or not self.barbs:
                K = 70
                ix = (slice(K//2, None, K), slice(K//2, None, K))
                self.barbs = m.barbs(
                    self.block.lons[ix], self.block.lats[ix],
                    self.block.x[ix], self.block.y[ix],
                    self.block.data[ix],
                    latlon=True, zorder=7, length=4, linewidth=0.5,
                )

    def click(self, lon, lat):
        self.selected = self.block.contains(lat, lon)
        return self.selected


class MPLCanvas(FigureCanvas):

    def __init__(self, parent=None, width=20, height=16, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        self.scale = 2.7e6
        self.pos = (15.00, 63.06)
        self._projection = 'aeqd'
        self._resolution = 'i'
        self.blocks = []
        self.selected = []

        self.mouse_origin = None

        super(MPLCanvas, self).__init__(fig)
        self.setParent(parent)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

        self.compute()

        self.fig = fig

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        self._projection = value
        self.full_refresh()

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value
        self.full_refresh()

    def add_block(self, block, draw=False, selected=False):
        self.blocks.append(BlockDrawer(block, draw, selected))

    def remove_blocks(self):
        self.blocks = []

    def compute(self):
        self.axes.clear()

        kwargs = {
            'projection': self.projection,
            'resolution': self.resolution,
            'lat_0': self.pos[1],
            'lon_0': self.pos[0],
            'ax': self.axes,
        }

        if PROJECTIONS[self.projection].method == 'width':
            kwargs.update({
                'width': self.scale * self.width() // self.height(),
                'height': self.scale,
            })
        else:
            # All the non-width projections are disabled for now
            circ = 2 * np.pi * EARTH_RADIUS * np.cos(self.pos[1] / 180 * np.pi)
            dlong = 180 * self.scale / circ
            dlat = 90 * self.scale / np.pi / EARTH_RADIUS * self.height() // self.width()
            kwargs.update({
                'llcrnrlon': self.pos[0] - dlong,
                'urcrnrlon': self.pos[0] + dlong,
                'llcrnrlat': max(self.pos[1] - dlat, -90),
                'urcrnrlat': min(self.pos[1] + dlat, 90),
            })

        m = Basemap(**kwargs)
        m.drawcoastlines()
        m.shadedrelief()
        # m.drawcountries(linestyle='dashed')
        # m.drawmapboundary(fill_color='aqua')
        # m.fillcontinents(color='coral', lake_color='aqua')
        m.drawparallels(np.arange(-80,81,10))
        m.drawmeridians(np.arange(-180,180,10))

        for b in self.blocks:
            b.draw(m, clear=True)

        self.m = m

    def full_refresh(self):
        self.compute()
        self.draw()

    def partial_refresh(self):
        for b in self.blocks:
            b.draw(self.m)
        self.draw()

    def zoom(self, amount=1):
        self.scale *= 1.1**amount
        self.full_refresh()

    def block_options(self):
        for b in self.selected:
            b.draw_field = True
        self.partial_refresh()

    def wheelEvent(self, event):
        self.zoom(-1 if event.angleDelta().y() > 0 else 1)

    def mouseDoubleClickEvent(self, event):
        print('double click')

    def mousePressEvent(self, event):
        self.mouse_origin = (event.x(), event.y())

    def mouseReleaseEvent(self, event):
        position = (event.x(), event.y())
        inv = self.axes.transData.inverted()
        coords = inv.transform((event.x(), self.height() - event.y()))

        # Right click => move center
        if position == self.mouse_origin and event.button() == QtCore.Qt.RightButton:
            self.pos = self.m(*coords, inverse=True)
            self.full_refresh()

        # Left click => select block
        elif position == self.mouse_origin and event.button() == QtCore.Qt.LeftButton:
            latlon = self.m(*coords, inverse=True)
            self.selected = []
            for block in self.blocks:
                if block.click(*latlon):
                    self.selected.append(block)
            self.partial_refresh()

        # Left drag => zoom in on area
        elif position != self.mouse_origin and event.button() == QtCore.Qt.LeftButton:
            old = inv.transform((self.mouse_origin[0], self.height() - self.mouse_origin[1]))
            self.scale = abs(coords[1] - old[1])
            self.pos = self.m((old[0] + coords[0])/2, (old[1] + coords[1])/2, inverse=True)
            self.full_refresh()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            inv = self.axes.transData.inverted()
            old = inv.transform((self.mouse_origin[0], self.height() - self.mouse_origin[1]))
            new = inv.transform((event.x(), self.height() - event.y()))
            lines = self.m.plot([old[0], new[0], new[0], old[0], old[0]],
                                [old[1], old[1], new[1], new[1], old[1]],
                                color='black', linestyle='dashed')
            self.draw()
            lines[0].remove()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('GMesh')

        main_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(main_widget)
        canvas = MPLCanvas(main_widget, width=5, height=4, dpi=100)
        layout.addWidget(canvas)

        file_menu = QtWidgets.QMenu('&File', self)
        file_menu.addAction('&Open file', self.open_file).setShortcut(QtGui.QKeySequence("Ctrl+O"))
        file_menu.addAction('Open &network', self.open_net).setShortcut(QtGui.QKeySequence("Ctrl+N"))
        self.menuBar().addMenu(file_menu)

        map_menu = QtWidgets.QMenu('&Map', self)
        projection_menu = QtWidgets.QMenu('&Projection', self)
        projection_grp = QtWidgets.QActionGroup(self)
        for key, proj in PROJECTIONS.items():
            act = projection_menu.addAction(proj.name, lambda key=key: self.projection(key))
            act.setCheckable(True)
            act.setChecked(key == 'aeqd')
            projection_grp.addAction(act)
        projection_grp.setExclusive(True)
        map_menu.addMenu(projection_menu)

        resolution_menu = QtWidgets.QMenu('&Resolution', self)
        resolution_grp = QtWidgets.QActionGroup(self)
        for key, name in [('c', 'Crude'), ('l', 'Low'), ('i', 'Intermediate'),
                          ('h', 'High'), ('f', 'Full')]:
            act = resolution_menu.addAction(name, lambda key=key: self.resolution(key))
            act.setCheckable(True)
            act.setChecked(key == 'c')
            resolution_grp.addAction(act)
        resolution_grp.setExclusive(True)
        map_menu.addMenu(resolution_menu)

        self.menuBar().addMenu(map_menu)
        self.setCentralWidget(main_widget)

        self.canvas = canvas

    def projection(self, value):
        self.canvas.projection = value

    def resolution(self, value):
        self.canvas.resolution = value

    def open_file(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setNameFilters(["HMs files (*.hms)",
                               "DEM files (*.dem)",
                               "NetCDF files (*.nc)"])
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialog.exec_():
            for fn in dialog.selectedFiles():
                for block in data.read(fn):
                    self.canvas.add_block(block)
            self.canvas.partial_refresh()

    def open_net(self):
        dates = ['01{:0>2}'.format(d) for d in range(5, 32)]
        dates.extend('02{:0>2}'.format(d) for d in range(1, 15))

        base = 'http://thredds.met.no/thredds/dodsC/fsiwt/AM25_Coupled2W_2015/netcdf/AM25_Coupled2W_2015{}00.nc'
        frameno = 0
        for d in dates:
            self.canvas.remove_blocks()
            fn = base.format(d)
            block = next(data.read(fn))
            self.canvas.add_block(block, True, True)
            for t in range(24):
                fn = 'frame{:0>3}.png'.format(frameno)
                frameno += 1
                print(d, t, fn)
                if exists(fn):
                    continue
                block.time = t
                self.canvas.full_refresh()
                self.canvas.fig.savefig(fn, dpi=200)

    def read_file(self, *args, **kwargs):
        print(args, kwargs)

    def keyPressEvent(self, event):
        if event.text() == 'q':
            self.close()
        elif event.text() == 'c':
            self.canvas.block_options()
        elif event.text() in {'+', '='}:
            self.canvas.zoom(-1)
        elif event.text() == '-':
            self.canvas.zoom()
        elif event.text() == 'r':
            self.canvas.partial_refresh()
        elif event.text() == 'R':
            self.canvas.full_refresh()
        else:
            print('Uncaught event: "{}"'.format(event.text()))


def run():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
