# label_lines
Automatically labels lines with rotation in Matplotlib

Primarily taken from https://pypi.org/project/matplotlib-label-lines/ but added collision avoidance
~~~~

Example:
labelLines(lines, 
        align=True, 
        zorder=2.5, 
        xvals = np.linspace(xmin+0.8*xrange, xmin+0.7*xrange, np.size(lines)), 
        check_collision=True,  
        filter_length = 3, 
        ha = 'left', 
        va='bottom', 
        bbox=dict(pad = 0, facecolor='none', edgecolor='none'), 
        backgroundcolor = [1,1,1,0], 
        rotation_mode = 'anchor', 
        fontsize = 8, 
        textcoords = 'offset points',
        xytext = (0,2.0)
    ) 

Label all lines with their respective legends.

    Parameters
    ----------
    lines : list of matplotlib lines
       The lines to label
    align : boolean, optional
       If True, the label will be aligned with the slope of the line at the location of the label. If False, they will be horizontal.
    xvals : (xfirst, xlast) or array of float, optional
       The location of the labels. If a tuple, the labels will be evenly spaced between xfirst and xlast (in the axis units).
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend do not use it anymore.
    align_length : int, optional
        Number of points to use for calculating alignment angle.
    kwargs : dict, optional
       Optional arguments passed to ax.text
