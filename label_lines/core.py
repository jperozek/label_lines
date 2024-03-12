from math import atan2, degrees
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import date2num, DateConverter, num2date
from matplotlib.container import ErrorbarContainer
from datetime import datetime
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import savgol_filter
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from scipy.interpolate import interp1d
from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString


class RotationAwareAnnotation2(mtext.Annotation):
    def __init__(self, s, xy, p, pa=None, ax=None, line=None, **kwargs):
        '''
            xy: first point
            p: second point to align to
            pa: first point to align to (same as xy...)
        '''
        self.rotation_on = True
        self.ax = ax or plt.gca()
        self.fig = ax.get_figure()
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
        
        # Get the length of the text box and re-calculate alignment point
        self.fig.draw_without_rendering()
        self.bb = self.get_window_extent()
        self.bb_data = ax.transData.inverted().transform_bbox(self.bb)
        self.dx = self.bb.x1 - self.bb.x0
        self.length = np.sqrt((self.bb.x1-self.bb.x0)**2 + (self.bb.y1-self.bb.y0)**2)
        self.p = self.find_end_pt()

    def find_end_pt(self):
        x1, y1 = self.xy
        x = self.line.get_xdata()
        y = self.line.get_ydata()
        f_interp = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        xscale = self.ax.get_xscale()
        if xscale == 'linear':
            x = np.linspace(x.min(), x.max(), 200)
        else:
            x = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y = f_interp(x)
        # find start index
        idx = np.argmin(np.abs(x-x1))
        x_pix, y_pix = self.ax.transData.transform(np.vstack([x[idx:], y[idx:]]).T).T
        dist = [((xp - x_pix[0])**2 + (yp - y_pix[0])**2)**0.5 for xp, yp in zip(x_pix, y_pix)]
        end_idx = np.argmin(np.abs(self.length - dist))
        x2_pix = x_pix[end_idx]
        y2_pix = y_pix[end_idx]
        x2, y2 = self.ax.transData.inverted().transform([x2_pix, y2_pix])
        return [x2, y2]


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

def get_textbox_poly(ax, tb):
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
    return poly1

def is_collision(ax, lines, other_tbs, tb, hit_tol=1):
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
   
    poly1 = get_textbox_poly(ax, tb)
     
    #Check if any lines fall inside
    p1 = Polygon(poly1.get_xy())
    tb_line = tb.line
    collision = False
    for line in lines:
        if line != tb_line:
            try:
                line = LineString(line.get_xydata())
                if p1.intersects(line):
                    collision = True
                    return collision
            except:
                pass
    #Check if any other labels fall inside other label boundaries
    for otb in other_tbs:
        p2 = Polygon(otb.get_xy())
        if p1.intersects(p2):
            collision = True
            return collision
        
    # Check if outside axis
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = p1.exterior.coords.xy
    if np.any(xx > xlim[1]) or np.any(xx < xlim[0]):
        collision = True
        return collision
    if np.any(yy > ylim[1]) or np.any(yy < ylim[0]):
        collision = True
        return collision
    
    # Uncomment To see where the hitboxes are   
    # ax.add_patch(poly1) 
    

    return collision


# Label line with line2D label data
def labelLine(line, x, label=None, align=True, drop_label=False, filter_length = 1, align_length=1, **kwargs):
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
    f_interp = interp1d(xdata, ydata, kind='quadratic', fill_value='extrapolate')
    xscale = ax.get_xscale()
    if xscale == 'linear':
        xdata = np.linspace(xdata.min(), xdata.max(), 200)
    else:
        xdata = np.logspace(np.log10(xdata.min()), np.log10(xdata.max()), 200)
    ydata = f_interp(xdata)

    #filter data
    if filter_length > 1:
        ydata = savgol_filter(ydata, filter_length, polyorder=2)

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception('The line %s only contains nan!' % line)

    # Make sure x is within range. Place at end if not
    if (not (np.min(xdata) < x < np.max(xdata))):
        x = xdata[-2]
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
    try: xfb = x_to_float(xdata[i+align_length])
    except: xfb = x_to_float(xdata[-1])
    ya = ydata[i]
    try: yb = ydata[i+align_length]
    except: yb = ydata[-1]
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
        
def labelLines(lines, align=True, xvals=None, check_collision=False, drop_label=False, filter_length=1, align_length=1, hit_tol=1, **kwargs):
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
        xdata = line.get_xdata()
        if len(xdata) < 2:
            continue
        if (not (np.min(xdata) < x < np.max(xdata))):
            continue
        textboxes.append(labelLine(line, x, label, align, drop_label, filter_length=filter_length, align_length=align_length, **kwargs))

    txt_boundaries = [get_textbox_poly(ax, tb) for tb in textboxes]
    other_boundaries = []
    if check_collision:
        for i, textbox in enumerate(textboxes):
            # other_boundaries = [bound for j, bound in enumerate(txt_boundaries) if j != i]
            if is_collision(ax, lines, other_boundaries, textbox, hit_tol):
                textbox.set_visible(False)
            other_boundaries += [get_textbox_poly(ax, textbox)]

    return textboxes


