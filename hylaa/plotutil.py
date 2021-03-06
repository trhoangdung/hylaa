'''
Hylaa Plot Utilities
Stanley Bak
Sept 2016
'''

import time
import random
import traceback
from collections import OrderedDict

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections
from matplotlib.path import Path
from matplotlib import colors
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

import numpy as np

from hylaa.file_io import write_matlab
from hylaa.starutil import AggregationParent, ContinuousPostParent
from hylaa.timerutil import Timers
from hylaa.containers import PlotSettings
from hylaa.util import Freezable
from hylaa.glpk_interface import LpInstance

def lighter(rgb_col):
    'return a lighter variant of an rgb color'

    return [(2.0 + val) / 3.0 for val in rgb_col]

def darker(rgb_col):
    'return a darker variant of an rgb color'

    return [val / 1.2 for val in rgb_col]

class AxisLimits(object):
    'container object for the plot axis limits'

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

class ModeColors(Freezable):
    'maps mode names -> colors'

    def __init__(self):
        self.init_colors()

        self.mode_to_color = {} # map mode name -> color string
        self.freeze_attrs()

    def init_colors(self):
        'initialize all_colors'

        self.all_colors = []

        # remove any colors with 'white' or 'yellow in the name
        skip_colors_substrings = ['white', 'yellow']
        skip_colors_exact = ['black', 'red', 'blue']

        for col in colors.cnames:
            skip = False

            for col_substring in skip_colors_substrings:
                if col_substring in col:
                    skip = True
                    break

            if not skip and not col in skip_colors_exact:
                self.all_colors.append(col)

        # we'll re-add these later; remove them before shuffling
        first_colors = ['lime', 'cyan', 'orange', 'magenta', 'green']

        for col in first_colors:
            self.all_colors.remove(col)

        # deterministic shuffle of all remaining colors
        random.seed(0)
        random.shuffle(self.all_colors)

        # prepend first_colors so they get used first
        self.all_colors = first_colors + self.all_colors

    def get_edge_face_colors(self, mode_name):
        '''
        get the edge and face colors from a mode name

        returns a tuple: (face_col, edge_col)
        '''

        col_name = self.mode_to_color.get(mode_name)

        if col_name is None:
            # pick the next color and save it
            col_name = self.all_colors[len(self.mode_to_color) % len(self.all_colors)]
            self.mode_to_color[mode_name] = col_name

        edge_col = colors.colorConverter.to_rgb(col_name)

        # make the faces a little lighter
        face_col = lighter(edge_col)

        return (face_col, edge_col)

class DrawnShapes(Freezable):
    'maintains shapes to be drawn'

    def __init__(self, plotman):
        self.plotman = plotman
        self.axes = plotman.axes
        self.mode_colors = plotman.mode_colors

        # create a blank invariant violation polys
        self.inv_vio_polys = collections.PolyCollection([], animated=True, alpha=0.7, edgecolor='red', facecolor='red')
        self.axes.add_collection(self.inv_vio_polys)

        # create a blank currently-tracked set of states poly
        self.cur_state_line2d = Line2D([], [], animated=True, color='k', lw=2, mew=2, ms=5, fillstyle='none')
        self.axes.add_line(self.cur_state_line2d)

        self.parent_to_polys = OrderedDict()
        self.parent_to_markers = OrderedDict()

        self.waiting_list_mode_to_polys = OrderedDict()
        self.aggregation_mode_to_polys = OrderedDict()

        self.trace = collections.LineCollection(
            [[(0, 0)]], animated=True, colors=('k'), linewidths=(3), linestyle='dashed')
        self.axes.add_collection(self.trace)

        if plotman.settings.extra_lines is not None:
            lines = plotman.settings.extra_lines
            self.extra_lines_col = collections.LineCollection(
                lines, animated=True, colors=('gray'), linewidths=(2), linestyle='dashed')
            self.axes.add_collection(self.extra_lines_col)
        else:
            self.extra_lines_col = None

        self.freeze_attrs()

    def get_artists(self, waiting_list):
        'get the list of artists, to be returned by animate function'

        rv = []

        for polys in self.parent_to_polys.values():
            rv.append(polys)

        for markers in self.parent_to_markers.values():
            rv.append(markers)

        self.set_waiting_list_polys(waiting_list)

        for polys in self.waiting_list_mode_to_polys.values():
            rv.append(polys)

        for polys in self.aggregation_mode_to_polys.values():
            rv.append(polys)

        rv.append(self.inv_vio_polys)

        if self.extra_lines_col:
            rv.append(self.extra_lines_col)

        rv.append(self.trace)

        rv.append(self.cur_state_line2d)

        return rv

    def set_waiting_list_polys(self, waiting_list):
        'set the polys from the waiting list'

        #print ".plotutil set_waiting_list_polys called... waitinglist ="
        #waiting_list.print_stats()

        self.clear_waiting_list_polys()

        # add deaggregated
        for star in waiting_list.deaggregated_list:
            verts = star.verts()
            self.add_waiting_list_poly(verts, star.mode.name)

        # add aggregated
        for star in waiting_list.aggregated_mode_to_state.values():
            verts = star.verts()

            self.add_aggregation_poly(verts, star.mode.name)

            # also show the sub-stars
            if isinstance(star.parent, AggregationParent):
                for substar in star.parent.stars:
                    verts = substar.verts()
                    self.add_waiting_list_poly(verts, substar.mode.name)

    def add_aggregation_poly(self, poly_verts, mode_name):
        '''add a polygon that's an aggregation on the waiting list'''

        polys = self.aggregation_mode_to_polys.get(mode_name)

        if polys is None:
            _, edge_col = self.mode_colors.get_edge_face_colors(mode_name)
            edge_col = darker(edge_col)

            polys = collections.PolyCollection([], lw=4, animated=True,
                                               edgecolor=edge_col, facecolor='none')
            self.axes.add_collection(polys)
            self.aggregation_mode_to_polys[mode_name] = polys

        paths = polys.get_paths()

        codes = [Path.MOVETO] + [Path.LINETO] * (len(poly_verts) - 2) + [Path.CLOSEPOLY]
        paths.append(Path(poly_verts, codes))

    def add_waiting_list_poly(self, poly_verts, mode_name):
        '''add a polygon on the waiting list'''

        polys = self.waiting_list_mode_to_polys.get(mode_name)

        if polys is None:
            face_col, edge_col = self.mode_colors.get_edge_face_colors(mode_name)

            polys = collections.PolyCollection([], lw=2, animated=True, alpha=0.3,
                                               edgecolor=edge_col, facecolor=face_col)
            self.axes.add_collection(polys)
            self.waiting_list_mode_to_polys[mode_name] = polys

        paths = polys.get_paths()

        codes = [Path.MOVETO] + [Path.LINETO] * (len(poly_verts) - 2) + [Path.CLOSEPOLY]
        paths.append(Path(poly_verts, codes))

    def clear_waiting_list_polys(self):
        'clears all the polygons drawn representing the waiting list'

        for polys in self.waiting_list_mode_to_polys.values():
            polys.get_paths()[:] = []

        for polys in self.aggregation_mode_to_polys.values():
            polys.get_paths()[:] = []

    def add_trace(self, trace_pts):
        'add to the current frame counter-example trace'

        paths = self.trace.get_segments()

        pts = np.array(trace_pts, dtype=float)

        val = [pts] + paths
        self.trace.set_segments(val)

        segs = self.trace.get_segments()

        # matplotlib does interpolation for larger lines, make sure it was disabled
        assert len(segs[0]) == len(pts), "matplotlib interpolation was not disabled"

    def reset_temp_polys(self):
        '''
        clear cur_state and invariant violation polygons. call at each step.
        '''

        self.inv_vio_polys.get_paths()[:] = []

        self.set_cur_state(None)

        self.trace.set_paths([])

    def set_cur_state(self, verts):
        'set the currently tracked set of states for one frame'

        l = self.cur_state_line2d

        if verts is None:
            l.set_visible(False)
        else:
            l.set_visible(True)

            if len(verts) <= 2:
                l.set_marker('o')
            else:
                l.set_marker(None)

            xdata = [x for x, _ in verts]
            ydata = [y for _, y in verts]

            l.set_xdata(xdata)
            l.set_ydata(ydata)

    def add_inv_vio_poly(self, poly_verts):
        'add an invariant violation polygon'

        paths = self.inv_vio_polys.get_paths()

        codes = [Path.MOVETO] + [Path.LINETO] * (len(poly_verts) - 2) + [Path.CLOSEPOLY]
        paths.append(Path(poly_verts, codes))

    def del_reachable_polys_from_parent(self, parent):
        '''
        stop drawing all polygons which were reached from a star and previously-
        added with add_reachable_poly_from_star
        '''

        assert isinstance(parent, ContinuousPostParent)

        polys = self.parent_to_polys.pop(parent, None)

        # polys may be none if it was an urgent transition
        if polys is not None:
            polys.remove() # reverses axes.add_collection
            polys.get_paths()[:] = []

        markers = self.parent_to_markers.pop(parent, None)

        if markers is not None:
            markers.remove()
            markers.set_xdata([])
            markers.set_ydata([])

    def thin_reachable_set(self):
        '''thin our the drawn reachable set to have less polygons (drawing optimization)'''

        for poly_col in self.parent_to_polys.values():
            paths = poly_col.get_paths()

            keep = True
            new_paths = []

            for p in paths:
                if keep:
                    new_paths.append(p)
    
                keep = not keep

            paths[:] = new_paths

        for line in self.parent_to_markers.values():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            new_xdata = []
            new_ydata = []

            keep = True

            for i in xrange(len(xdata)):
                if keep:
                    new_xdata.append(xdata[i])
                    new_ydata.append(ydata[i])

                keep = not keep

            line.set_xdata(new_xdata)
            line.set_ydata(new_ydata)

    def add_reachable_poly(self, poly_verts, parent, mode_name):
        '''add a polygon which was reachable'''

        assert isinstance(parent, ContinuousPostParent)

        if len(poly_verts) <= 2:
            markers = self.parent_to_markers.get(parent)

            if markers is None:
                face_col, edge_col = self.mode_colors.get_edge_face_colors(mode_name)

                markers = Line2D([], [], animated=True, ls='None', alpha=0.5, marker='o', mew=2, ms=5,
                                 mec=edge_col, mfc=face_col)
                self.axes.add_line(markers)
                self.parent_to_markers[parent] = markers

            xdata = markers.get_xdata()
            ydata = markers.get_ydata()
            xdata.append(poly_verts[0][0])
            ydata.append(poly_verts[0][1])
            markers.set_xdata(xdata)
            markers.set_ydata(ydata)
        else:
            polys = self.parent_to_polys.get(parent)

            if polys is None:
                face_col, edge_col = self.mode_colors.get_edge_face_colors(mode_name)
                polys = collections.PolyCollection([], lw=2, animated=True, alpha=0.5,
                                                   edgecolor=edge_col, facecolor=face_col)
                self.axes.add_collection(polys)
                self.parent_to_polys[parent] = polys

            paths = polys.get_paths()

            codes = [Path.MOVETO] + [Path.LINETO] * (len(poly_verts) - 2) + [Path.CLOSEPOLY]
            paths.append(Path(poly_verts, codes))

class InteractiveState(object):
    'container object for interactive plot state'

    def __init__(self):
        self.paused = False
        self.step = False

class PlotManager(Freezable):
    'manager object for plotting during or after computation'

    def __init__(self, hylaa_engine, plot_settings):
        assert isinstance(plot_settings, PlotSettings)

        # matplotlib default rcParams caused incorrect trace output due to interpolation
        rcParams['path.simplify'] = False

        self.engine = hylaa_engine
        self.settings = plot_settings

        self.fig = None
        self.axes = None
        self.actual_limits = None # AxisLimits object
        self.drawn_limits = None # AxisLimits object

        self.mode_colors = ModeColors()
        self.shapes = None # instance of DrawnShapes
        self.interactive = InteractiveState()

        self.drew_first_frame = False # one-time flag
        self._anim = None # animation object

        if self.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE or \
           self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            self.settings.min_frame_time = 0.0 # for interactive or video plots, draw every frame

        self.cur_reachable_polys = 0 # number of polygons currently drawn
        self.draw_stride = plot_settings.draw_stride # draw every 2nd poly, or every 4th, ect. (if over poly limit)
        self.draw_cur_step = 0 # the current poly in the step

        if self.settings.plot_mode == PlotSettings.PLOT_MATLAB:
            self.reach_poly_data = OrderedDict()

        self.freeze_attrs()

    def plot_trace(self, num_steps, sim_bundle, start_basis_matrix, basis_point):
        'plot a trace to a basis_point in a symbolic state'

        if self.shapes is not None and self.settings.plot_traces:
            pts = []

            for step in xrange(num_steps+1):
                basis_vec_list, sim_center = sim_bundle.get_vecs_origin_at_step(step, num_steps)

                if start_basis_matrix is None:
                    basis_matrix = basis_vec_list
                else:
                    basis_matrix = np.dot(start_basis_matrix, basis_vec_list)

                offset = np.dot(basis_matrix.T, basis_point)
                point = np.add(sim_center, offset)

                x = point[self.settings.xdim]
                y = point[self.settings.ydim]

                pts.append((x, y))

            self.shapes.add_trace(pts)

    def update_axis_limits(self, points_list):
        'update the axes limits to include the passed-in point list'

        first_draw = False

        if self.drawn_limits is None:
            first_draw = True
            first_x, first_y = points_list[0]
            self.actual_limits = AxisLimits()
            self.drawn_limits = AxisLimits()
            self.actual_limits.xmin = self.actual_limits.xmax = first_x
            self.actual_limits.ymin = self.actual_limits.ymax = first_y
            self.drawn_limits.xmin = self.drawn_limits.xmax = first_x
            self.drawn_limits.ymin = self.drawn_limits.ymax = first_y

        lim = self.actual_limits
        drawn = self.drawn_limits

        for p in points_list:
            x, y = p

            if x < lim.xmin:
                lim.xmin = x
            elif x > lim.xmax:
                lim.xmax = x

            if y < lim.ymin:
                lim.ymin = y
            elif y > lim.ymax:
                lim.ymax = y

        is_outside = lim.xmin < drawn.xmin or lim.xmax > drawn.xmax or lim.ymin < drawn.ymin or lim.ymax > drawn.ymax
        if first_draw or (is_outside and lim.xmin != lim.xmax and lim.ymin != lim.ymax):

            # expand drawn limits to surpass actual
            dx = lim.xmax - lim.xmin
            dy = lim.ymax - lim.ymin
            ratio = self.settings.extend_plot_range_ratio

            drawn.xmin = lim.xmin - dx * ratio
            drawn.xmax = lim.xmax + dx * ratio
            drawn.ymin = lim.ymin - dy * ratio
            drawn.ymax = lim.ymax + dy * ratio

            if drawn.xmin == drawn.xmax:
                self.axes.set_xlim(drawn.xmin - 1e-1, drawn.xmax + 1e-1)
            else:
                self.axes.set_xlim(drawn.xmin, drawn.xmax)

            if drawn.ymin == drawn.ymax:
                self.axes.set_ylim(drawn.ymin - 1e-1, drawn.ymax + 1e-1)
            else:
                self.axes.set_ylim(drawn.ymin, drawn.ymax)

    def reset_temp_polys(self):
        'clear the invariant violation polygons (called once per iteration)'

        if self.shapes is not None:
            self.shapes.reset_temp_polys()

    def add_inv_violation_star(self, star):
        'add an invariant violation region'

        if self.shapes is not None:
            verts = star.verts()

            self.shapes.add_inv_vio_poly(verts)

            if self.settings.label.axes_limits is None:
                self.update_axis_limits(verts)

    def create_plot(self):
        'create the plot'

        if self.settings.plot_mode != PlotSettings.PLOT_NONE and self.settings.plot_mode != PlotSettings.PLOT_MATLAB:
            self.fig, self.axes = plt.subplots(nrows=1, figsize=self.settings.plot_size)
            ha = self.engine.hybrid_automaton

            title = self.settings.label.title
            title = title if title is not None else ha.name

            x_label = self.settings.label.x_label
            x_label = x_label if x_label is not None else ha.variables[self.settings.xdim].capitalize()
            y_label = self.settings.label.y_label
            y_label = y_label if y_label is not None else ha.variables[self.settings.ydim].capitalize()

            self.axes.set_xlabel(x_label, fontsize=self.settings.label.label_size)
            self.axes.set_ylabel(y_label, fontsize=self.settings.label.label_size)
            self.axes.set_title(title, fontsize=self.settings.label.title_size)

            if self.settings.label.axes_limits is not None:
                # hardcoded axes limits
                xmin, xmax, ymin, ymax = self.settings.label.axes_limits

                self.axes.set_xlim(xmin, xmax)
                self.axes.set_ylim(ymin, ymax)

            if self.settings.grid:
                self.axes.grid(True)

            # make the x and y axis animated in case of rescaling
            self.axes.xaxis.set_animated(True)
            self.axes.yaxis.set_animated(True)

            plt.tick_params(axis='both', which='major', labelsize=self.settings.label.tick_label_size)
            plt.tight_layout()

            self.shapes = DrawnShapes(self)

    def del_parent_successors(self, parent):
        '''stop plotting a parent's's sucessors'''

        if self.shapes is not None:
            self.shapes.del_reachable_polys_from_parent(parent)

            # maybe we want to revert axis limits here?

    def state_popped(self):
        'called whenever a state is popped from the waiting list'

        self.draw_cur_step = 0 # reset the cur_step counter

    def add_reachable_poly_data(self, verts, mode_name):
        '''
        Add raw reachable poly data for use with certain plotting modes (matlab).
        '''

        data = self.reach_poly_data
        # values are a tuple (color, [poly1, poly2, ...])

        if mode_name not in data:
            ecol, fcol = self.mode_colors.get_edge_face_colors(mode_name)
            data[mode_name] = (ecol, fcol, [])
            
        data[mode_name][2].append(verts)

    def plot_current_star(self, star):
        '''
        plot the current SymbolicState according to the plot settings

        returns True if the plot was skipped (due to too many polyons on the screen
        '''

        skipped_plot = True

        if self.settings.plot_mode == PlotSettings.PLOT_MATLAB:
            verts = star.verts()

            self.add_reachable_poly_data(verts, star.mode.name)
        elif self.settings.plot_mode != PlotSettings.PLOT_NONE:
            Timers.tic("plot_current_star()")

            if self.draw_cur_step % self.draw_stride == 0:
                skipped_plot = False

                verts = star.verts()

                if self.settings.print_lp_at_each_step:
                    star.print_lp()

                self.shapes.set_cur_state(verts)

                if self.settings.label.axes_limits is None:
                    self.update_axis_limits(verts)

                # possibly thin out the reachable set of states
                max_polys = self.settings.max_shown_polys

                if max_polys > 0 and self.cur_reachable_polys >= max_polys:
                    self.shapes.thin_reachable_set()
                    self.cur_reachable_polys /= 2
                    self.draw_cur_step = 0
                    self.draw_stride *= 2

                self.cur_reachable_polys += 1

                self.shapes.add_reachable_poly(verts, star.parent, star.mode.name)

            self.draw_cur_step += 1

            Timers.toc("plot_current_star()")

        return skipped_plot

    def compute_and_animate(self, step_func, is_finished_func):
        'do the computation, plotting during the process'

        def anim_func(force_single_frame):
            'performs several steps of the computation and draws an animation frame'

            if not force_single_frame and self.interactive.paused:
                Timers.tic("paused")
                time.sleep(0.1)
                Timers.toc("paused")
            else:
                Timers.tic("frame")

                start_time = time.time()
                while not is_finished_func():
                    step_func()

                    # do several computation steps per frame if they're fast (optimization)
                    if force_single_frame or time.time() - start_time > self.settings.min_frame_time:
                        break

                # if we just wanted a single step
                if self.interactive.step:
                    self.interactive.step = False
                    self.interactive.paused = True

                if is_finished_func():
                    self.shapes.inv_vio_polys.set_visible(False)

                Timers.toc("frame")

                if self.interactive.paused and not force_single_frame:
                    print "Paused After Frame #{}".format(Timers.timers['frame'].num_calls)

            rv = self.shapes.get_artists(self.engine.waiting_list)

            rv += [self.axes.xaxis, self.axes.yaxis]

            return rv

        def init_func():
            'animation init function'

            rv = self.shapes.get_artists(self.engine.waiting_list)

            # it seems we only need to do this once...
            #if not self.drew_first_frame:
            #    self.drew_first_frame = True

            #    print "drew first frame"
            rv += [self.axes.xaxis, self.axes.yaxis]

            return rv

        def anim_iterator():
            'generator for the computation iterator'
            Timers.tic("total")

            # do the computation until its done
            while not is_finished_func():
                yield False

            # redraw one more (will clear cur_state)
            #yield False

            Timers.toc("total")

            LpInstance.print_stats()
            Timers.print_stats()

        def next_pressed(_):
            'event function for next button press'
            self.interactive.paused = False

        def step_pressed(_):
            'event function for step button press'
            self.interactive.paused = False
            self.interactive.step = True

        iterator = anim_iterator

        if self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
            if self.settings.video.frames is None:
                print "Warning: PLOT_VIDEO requires explicitly setting plot_settings.video.frames (default is 100)."
            else:
                iterator = self.settings.video.frames

        if self.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            # shrink plot, add buttons
            plt.subplots_adjust(bottom=0.12)

            axnext = plt.axes([0.81, 0.02, 0.1, 0.05])
            bnext = Button(axnext, 'Next', color='0.85', hovercolor='0.85')
            bnext.on_clicked(next_pressed)

            axstep = plt.axes([0.61, 0.02, 0.1, 0.05])
            bstep = Button(axstep, 'Step', color='0.85', hovercolor='0.85')
            bstep.on_clicked(step_pressed)

        # process a certain number of frames if the settings desire it
        for _ in xrange(self.settings.skip_frames):
            anim_func(True)

        self._anim = animation.FuncAnimation(self.fig, anim_func, iterator, init_func=init_func,
                                             interval=self.settings.anim_delay_interval, blit=True, repeat=False)

        if not self.settings.skip_show_gui:
            if self.settings.plot_mode == PlotSettings.PLOT_VIDEO:
                self.save_video(self._anim)
            elif self.settings.plot_mode == PlotSettings.PLOT_IMAGE:
                self.save_image()
            else:
                plt.show()

    def save_matlab(self):
        'save a matlab script'

        self.engine.run_to_completion()

        filename = self.settings.filename

        if filename is None:
            filename = "plot_reach.m"

        if not filename.endswith('.m'):
            filename = filename + '.m'

        write_matlab(filename, self.reach_poly_data, self.settings, self.engine.hybrid_automaton)

    def save_image(self):
        'save an image file'

        self.engine.run_to_completion()

        filename = self.settings.filename

        if filename is None:
            filename = "plot.png"

        plt.savefig(filename, bbox_inches='tight')

    def save_video(self, func_anim_obj):
        'save a video file of the given FuncAnimation object'

        filename = self.settings.filename

        if filename is None:
            filename = "video.avi" # mp4 is also possible

        fps = self.settings.video.fps
        codec = self.settings.video.codec

        print "Saving {} at {:.2f} fps using ffmpeg with codec '{}'.".format(
            filename, fps, codec)

        # if this fails do: 'sudo apt-get install ffmpeg'
        try:
            start = time.time()

            extra_args = []

            if codec is not None:
                extra_args += ['-vcodec', str(codec)]

            func_anim_obj.save(filename, fps=fps, extra_args=extra_args)

            dif = time.time() - start
            print "Finished creating {} ({:.2f} seconds)!".format(filename, dif)
        except AttributeError:
            traceback.print_exc()

            print "\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?"

def debug_plot_star(star, col='k-', lw=1):
    '''
    debug function for plotting a star. This calls plt.plot(), so it's up to you
    to call plt.show() afterwards
    '''

    verts = star.verts()

    # wrap polygon back to first point
    verts.append(verts[0])

    xs = [ele[0] for ele in verts]
    ys = [ele[1] for ele in verts]

    plt.plot(xs, ys, col, lw=lw)

# monkey patch function for blitting tick-labels
# see http://stackoverflow.com/questions/17558096/animated-title-in-matplotlib
def _blit_draw(_self, artists, bg_cache):
    'money-patch version of animation._blit_draw'
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

animation.Animation._blit_draw = _blit_draw



