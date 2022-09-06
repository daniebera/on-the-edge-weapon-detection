"""postprocess.py

Implementation of the intermediate processing between the people detection and the weapon detection.
"""


def compute_img_crops(orig_img_w, orig_img_h, max_dim, xcen, ycen):
    """Compute the offsets to obtain the image crop on a detected person.
    
    # Args
        orig_img_w: width of the original image
        orig_img_h: height of the original image
        max_dim: max side dimension among the target image and the bounding box heights and widths
        xcen: coordinate along the x-axis of the bounding box center
        ycen: coordinate along the y-axis of the bounding box center

    # Returns
        xstart: starting offset to make crop along the x-axis
        xend: ending offset to make crop along the x-axis
        ystart: starting offset to make crop along the y-axis
        yend: ending offset to make crop along the y-axis
    """
    xstart_ = max(0, xcen - max_dim/2)  # with negative x start from 0
    xslack = abs(min(0, xcen - max_dim/2))  # store negative x
    xend_ = min(orig_img_w-1, xcen + max_dim/2)  # with x exceeding img width end to img width (minus 1)
    xsurplus = max(0, xcen + max_dim/2 - (orig_img_w-1))  # store exceeding x

    xstart = xstart_ - xsurplus  # start with left shift (eventual)
    xend = xend_ + xslack  # end with right shift (eventual)

    ystart_ = max(0, ycen - max_dim/2)  # with negative y start from 0
    yslack = abs(min(0, ycen - max_dim/2))  # store negative y
    yend_ = min(orig_img_h-1, ycen + max_dim/2)  # with y exceeding img height end to img height (minus 1)
    ysurplus = max(0, ycen + max_dim/2 - (orig_img_h-1))  # store exceeding y

    ystart = ystart_ - ysurplus
    yend = yend_ + yslack

    return int(xstart),int(xend),int(ystart),int(yend)


class Box:
    """Box class useful to handle Bounding Boxes attributes."""

    def __init__(self, bb):
        xmn,ymn,xmx,ymx = bb

        # check if width is even
        if (xmx - xmn) % 2 != 0:
            xmx -= 1
        # check if height is even
        if (ymx - ymn) % 2 != 0:
            ymx -= 1

        self.xmin = xmn
        self.ymin = ymn
        self.xmax = xmx
        self.ymax = ymx

        # compute center (w.r.t. whole img), width, height
        self.xcen = ((xmx - xmn) / 2) + xmn
        self.ycen = ((ymx - ymn) / 2) + ymn
        self.w = xmx - xmn
        self.h = ymx - ymn

        self.label = -1

    def set_label(self, label):
        self.label = label
