from math import atan2, degrees
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import date2num, DateConverter, num2date
from matplotlib.container import ErrorbarContainer
from datetime import datetime
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms


class RotationAwareAnnotation2(mtext.Annotation):
    def __init__(self, s, xy, p, pa=None, ax=None, line=None, **kwargs):
        self.rotation_on = True
        self.ax = ax or plt.gca()
        self.p = p
        if not pa:
            self.pa = xy
        kwargs.update(rotation_mode=kwargs.get("rotation_mode", "anchor"))
        mtext.Annotation.__init__(self, s, xy, **kwargs)
        self.set_transform(mtransforms.IdentityTransform())
        if 'clip_on' in kwargs:
            self.set_clip_path(self.ax.patch)
        self.ax._add_text(self)
        if line:
            self.line = line

    def calc_angle(self):
        if self.rotation_on:
            p = self.ax.transData.transform_point(self.p)
            pa = self.ax.transData.transform_point(self.pa)
            ang = np.arctan2(p[1]-pa[1], p[0]-pa[0])
            return np.rad2deg(ang)
        else:
            return 0

    def _get_rotation(self):
        return self.calc_angle()

    def _set_rotation(self, rotation):
        pass

    _rotation = property(_get_rotation, _set_rotation)


def is_collision(ax, lines, tb, hit_tol=1):
    ''' Checks to see if the textbox collides with any data points
        for lines already passed. Very convoluted. 

        The goal is to use the contains_points method of a polygon patch,
        however we need to be careful about rotating in display coordinates
        so that there is no distortion from our axis not being square.

        Only work on initial drawing. Does not resize with plot. One day 
        this should be made into a property or the RotationAwareAnnotation 
        class, however it currently requires the figure to be redrawn,
        which takes too much time.  

        ax: axes object for figure
        lines: all lines being plotted
        tb: The textbox to see if a line intersects
        hit_tol: The number of points that are allowed to fall within the 
                hit box until a collision is detected
    '''
    # Need to render the figure first to get everything it its place...
    ax.figure.canvas.draw_idle()
    # Get the bounding box of the textbox without rotation (smalles size).
    tb.rotation_on = False
    bb = tb.get_window_extent(renderer = ax.figure.canvas.get_renderer())
    tb.rotation_on = True

    # Make a scaled dummy polygon to check if a point is inside
    x0 = bb.x0 
    y0 = bb.y0 
    width = (bb.x1 - bb.x0)
    height = (bb.y1 - bb.y0)
    x1 = x0 + width
    y1 = y0 + height

    pgon_disp = np.array([[x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [x0, y0]])

    # Rotate the polygon in display coordinates to avoid skewing in non-square
    # data coordinates
    rotate_transform = mpl.transforms.Affine2D().rotate_deg_around(x0, y0, tb.get_rotation())
    pgon_disp = rotate_transform.transform(pgon_disp)
    # Translate for the xy offset
    # translate_transform = mpl.transforms.Affine2D().translate(tb._x*65, tb._y*65)
    # pgon_disp = translate_transform.transform(pgon_disp)
    # Convert polygon to data coordinates
    transform_disp2data = ax.transData.inverted()
    ppgon_data = transform_disp2data.transform(pgon_disp)
    # Make it into a patch
    poly1 = mpl.patches.Polygon(ppgon_data, closed = True, alpha=0.3, color='red', linewidth = None)
    
     
    #Check if any lines fall inside
    tb_line = tb.line
    collision = False
    for line in lines:
        if line != tb_line:
            hit_array = poly1.contains_points(line.get_xydata())
            # if np.any(hit_array):
            if np.count_nonzero(hit_array)>hit_tol:
                collision = True
    
    # Uncomment To see where the hitboxes are   
    # ax.add_patch(poly1) 

    return collision



# Label line with line2D label data
def labelLine(line, x, label=None, align=True, drop_label=False, filter_length = 1, **kwargs):
    '''Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    #filter data
    ydata = uniform_filter1d(ydata, filter_length)

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception('The line %s only contains nan!' % line)

    # Find first segment of xdata containing x
    if len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= x <= max(xa, xb): #Looks at pairs of x[i] x[i+1] and stops when xval is between it
                break
        else:
            raise Exception('x label location is outside data range!')

    def x_to_float(x):
        """Make sure datetime values are properly converted to floats."""
        return date2num(x) if isinstance(x, datetime) else x

    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa) #Calculates y based on linear extrapolation?

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(("%s could not be annotated due to `nans` values. "
                       "Consider using another location via the `x` argument.") % line,
                      UserWarning)
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform((xfa, ya)) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    #ax.text(x, y, label, rotation=rotation, **kwargs)
    if align:
        txt = RotationAwareAnnotation2(label, xy=(xfa,ya), p = (xfb,yb), ax=ax, line = line, **kwargs)
    else:
        txt = ax.annotate(label, (x, y), rotation=rotation, **kwargs)
        
    return txt
        
def labelLines(lines, align=True, xvals=None, check_collision=False, drop_label=False, filter_length=1, hit_tol=1, **kwargs):
    '''Label all lines with their respective legends.

    Parameters
    ----------
    lines : list of matplotlib lines
       The lines to label
    align : boolean, optional
       If True, the label will be aligned with the slope of the line
       at the location of the label. If False, they will be horizontal.
    xvals : (xfirst, xlast) or array of float, optional
       The location of the labels. If a tuple, the labels will be
       evenly spaced between xfirst and xlast (in the axis units).
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    '''
    ax = lines[0].axes

    labLines, labels = [], []
    handles, allLabels = ax.get_legend_handles_labels()

    all_lines = []
    for h in handles:
        if isinstance(h, ErrorbarContainer):
            all_lines.append(h.lines[0])
        else:
            all_lines.append(h)

    # Take only the lines which have labels other than the default ones
    for line in lines:
        if line in all_lines:
            label = allLabels[all_lines.index(line)]
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xvals = ax.get_xlim()  # set axis limits as annotation limits, xvals now a tuple
        if type(xvals) == tuple:
            xmin, xmax = xvals
            xscale = ax.get_xscale()
            if xscale == "log":
                xvals = np.logspace(np.log10(xmin), np.log10(xmax), len(labLines)+2)[1:-1]
            else:
                xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]
    
            if isinstance(ax.xaxis.converter, DateConverter):
                # Convert float values back to datetime in case of datetime axis
                xvals = [num2date(x).replace(tzinfo=ax.xaxis.get_units())
                         for x in xvals]

    textboxes = []
    for line, x, label in zip(labLines, xvals, labels):
        textboxes.append(labelLine(line, x, label, align, drop_label, filter_length=filter_length, **kwargs))

    if check_collision:
        for textbox in textboxes:
            if is_collision(ax, lines, textbox, hit_tol):
                textbox.set_visible(False)

    return textboxes


