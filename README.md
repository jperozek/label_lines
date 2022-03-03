# label_lines
Automatically labels lines with rotation in Matplotlib

Primarily taken from https://pypi.org/project/matplotlib-label-lines/


Example:
labelLines(lines, 
        align=True, 
        zorder=2.5, 
        xvals = np.linspace(xmin+0.8*xrange, xmin+0.7*xrange, np.size(lines)), 
        check_collision=True,  
        hit_tol=20, 
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
