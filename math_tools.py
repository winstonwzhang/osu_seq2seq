# math helper functions

import numpy as np
import bisect
import math
from itertools import accumulate

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from utils import lazyval

import pdb

# modified from https://github.com/llllllllll/slider/blob/master/slider/curve.py
# thank goodness for open source osu code (T-T)ゞ

# curve classes

class Curve:
    """Curves are functions that define the path that a slider takes from start to stop.
    Parameters
    ----------
    points : list[Position]
        The positions of the control points.
    req_length : float
        The pixel length of the curve.
    """
    def __init__(self, points, req_length):
        self.points = points
        self.req_length = req_length

    @classmethod
    def from_kind_and_points(cls, kind, points, req_length):
        if kind == 'B':
            return MultiBezier(points, req_length)
        if kind == 'L':
            return Linear(points, req_length)
        if kind == 'C':
            return Catmull(points, req_length)
        if kind == 'P':
            if len(points) != 3:
                # it seems osu! uses the bezier curve if there are more than 3 points
                # https://github.com/ppy/osu/blob/7fbbe74b65e7e399072c198604e9db09fb729626/osu.Game/Rulesets/Objects/SliderCurve.cs#L32  # noqa
                return MultiBezier(points, req_length)

            try:
                center = get_center(*points)
            except ValueError:
                # we cannot use a perfect curve function for collinear points
                # osu! also falls back to a bezier here
                return MultiBezier(points, req_length)

            return Perfect(points, req_length, _center=center)

        raise ValueError(f'unknown curve kind: {kind!r}')


class Bezier(Curve):
    """Bezier curves of degree ``len(points)``.
    Parameters
    ----------
    points : list[Position]
        The positions of the control points.
    req_length : float
        The pixel length of the curve.
    """
    def __init__(self, points, req_length):
        super().__init__(points, req_length)
        self._coordinates = np.array(points).T

    def __call__(self, t, request=None):
        # return x and y position of parameter t
        if request is None:
            coordinates = self.at(t * (self.req_length / self.length))
            return [coordinates[0, 0], coordinates[0, 1]]
        # return slope of two input points
        elif request == 'deriv':
            _, dir = getVector(self(t[0]),self(t[1]))
            return dir

    def at(self, t):
        """Compute the positions of the bezier curve for a sequence of times in [0, 1].
        Parameters
        ----------
        t : np.ndarray
            The times to compute for.
        Returns
        -------
        coordinates : np.ndarray
            A two dimensional array of shape (len(t), 2) where the first column
            holds the x coordinate at time t and the second column holds the y
            coordinate at time t.
        """
        t = np.asarray(t).reshape(-1, 1)

        points = self.points

        n = len(points) - 1
        ixs = np.arange(n + 1)
        return np.sum(
            (
                comb(n, ixs) *
                (1 - t) ** (n - ixs) *
                t ** ixs
            )[:, np.newaxis] *
            self._coordinates,
            axis=-1,
        )
    
    @lazyval
    def length(self):
        """Approximates length as piecewise linear function.
        """
        # NOTE: if the error is high, try increasing the samples
        points = self.at(np.linspace(0, 1, 50))
        return np.sum(
            np.sqrt(
                np.sum(
                    np.square(
                        np.diff(points, axis=0),
                    ),
                    axis=1,
                ),
            ),
        )


class _MetaCurveMixin:
    """Mixin for defining curves which are collections of other curves.
    Implementers must provide a ``_curves`` attribute which is the collection
    of curves.
    """
    @lazyval
    def _ts(self):
        lengths = [c.length for c in self._curves]
        length = sum(lengths)
        out = []
        for i, j in enumerate(accumulate(lengths[:-1])):
            self._curves[i].req_length = lengths[i]
            out.append(j / length)
        self._curves[-1].req_length = max(
            0,
            lengths[-1] - (length - self.req_length),
        )
        out.append(1)
        return out

    def __call__(self, t, request=None):
        if request is None:
            ts = self._ts
            if len(self._curves) == 1:
                # Special case where we only have one curve
                return self._curves[0](t)

            bi = bisect.bisect_left(ts, t)
            if bi == 0:
                pre_t = 0
            else:
                pre_t = ts[bi - 1]

            post_t = ts[bi]

            return self._curves[bi]((t - pre_t) / (post_t - pre_t))
        else:
            _, dir = getVector(self(t[0]),self(t[1]))
            return dir


class MultiBezier(_MetaCurveMixin, Curve):
    """MultiBezier is a collection of `:class:~slider.curve.Bezier` curves
    stitched together at the ends.
    Parameters
    ----------
    points : list[Position]
        The positions of the control points.
    req_length : float
        The pixel length of the curve.
    """
    def __init__(self, points, req_length):
        super().__init__(points, req_length)

        self._curves = [
            Bezier(subpoints, None)
            for subpoints in self._split_at_dupes(points)
        ]

    @staticmethod
    def _split_at_dupes(input_):
        """Split a sequence of points on duplicates.
        Parameters
        ----------
        inp : list[any]
            The input sequence to split.
        Yields
        ------
        group : list[any]
            The groups split on duplicates.
        """
        old_ix = 0
        for n, (a, b) in enumerate(zip(input_, input_[1:]), 1):
            if a == b:
                yield input_[old_ix:n]
                old_ix = n

        tail = input_[old_ix:]
        if tail:
            yield tail


class Linear(_MetaCurveMixin, Curve):
    """Linear curves traverse the line segments between the given points.
    Parameters
    ----------
    points : list[Position]
        The positions of the control points.
    req_length : float
        The pixel length of the curve.
    """
    def __init__(self, points, req_length):
        super().__init__(points, req_length)

        self._curves = [
            Bezier(subpoints, None) for subpoints in zip(points, points[1:])
        ]


class Perfect(Curve):
    """Perfect curves traverse the circumference of the circle that passes
    through the given points.
    Parameters
    ----------
    points : list[Position]
        The positions of the control points.
    req_length : float
        The pixel length of the curve.
    """
    def __init__(self, points, req_length, *, _center=None):
        # NOTE: _center exists as an optimization to not recompute the center
        # of the circle when calling ``Curve.from_kind_and_points``.
        self.points = points
        self.req_length = req_length

        if _center is None:
            _center = get_center(*points)

        self._center = _center

        coordinates = np.array(points) - _center

        # angles of 3 points to center
        start_angle, end_angle = np.arctan2(
            coordinates[::2, 1],
            coordinates[::2, 0],
        )

        # normalize so that self._angle is positive
        if end_angle < start_angle:
            end_angle += 2 * math.pi

        # angle of arc sector that describes slider
        self._angle = end_angle - start_angle

        # switch angle direction if necessary
        a_to_c = coordinates[2] - coordinates[0]
        ortho_a_to_c = np.array((a_to_c[1], -a_to_c[0]))
        if np.dot(ortho_a_to_c, coordinates[1] - coordinates[0]) < 0:
            self._angle = -(2 * math.pi - self._angle)

        length = abs(
            self._angle *
            math.sqrt(coordinates[0][0] ** 2 + coordinates[0][1] ** 2),
        )
        if length > req_length:
            self._angle *= req_length / length

    def __call__(self, t, request=None):
        if request is None:
            return rotate(self.points[0], self._center, self._angle * t)
        else:
            _, dir = getVector(self(t[0]),self(t[1]))
            return dir


class Catmull(Curve):
    """Catmull curves implement the Catmull-Rom spline algorithm for defining
    the path.
    """
    def __init__(self, points, req_length):
        super().__init__(points, req_length)
        points = np.array(points)

        # implementation follows notes at https://cubic.org/docs/hermite.htm
        self.h = np.array([[2,  -2,  1,  1],
                           [-3,  3, -2, -1],
                           [0,   0,  1,  0],
                           [1,   0,  0,  0]])

        tangents_x = []
        tangents_y = []
        self.Cxs = []
        self.Cys = []
        # The tangent for point i is defined as 0.5 * (P_(i + 1) - P_(i - 1)),
        # so we need to consider the point behind it and the point in front of
        # it. We:
        # * roll points right by one. Then we replace first element with
        #   the previous first element (which is now second) so the first
        #   tangent can be calculated properly.
        # * roll points left by one. Then we replace last element with the
        #   previous last element (which is now second to last) so the last
        #   tangent can be calculated properly.
        # to create the p_aheads and p_behinds lists respectively.
        # For example, in a list of five points, we have:
        # * points =    [p1, p2, p3, p4, p5]
        # * p_aheads =  [p2, p3, p4, p5, p5]
        # * p_behinds = [p1, p1, p2, p3, p4]

        p_aheads = np.roll(points, 1)
        p_aheads[0] = p_aheads[1]

        p_behinds = np.roll(points, -1)
        p_behinds[-1] = p_behinds[-2]

        # we interpolate x and y separately, so track their tangents in two
        # separate lists.
        for (p_ahead, p_behind) in zip(p_aheads, p_behinds):
            tangent = 0.5 * (p_ahead[0] - p_behind[0])
            tangents_x.append(tangent)

            tangent = 0.5 * (p_ahead[1] - p_behind[1])
            tangents_y.append(tangent)

        # for each curve we consider its start and end point (and start and end
        # tangent). This means the number of curves will be one less than the
        # number of points.
        for ((p1, p2), (t1, t2)) in zip(zip(points, points[1:]),
                                        zip(tangents_x, tangents_x[1:])):
            Cx = np.array([p1[0], p2[0], t1, t2])
            # make it a column vector
            Cx = Cx[:, np.newaxis]
            self.Cxs.append(Cx)

        for ((p1, p2), (t1, t2)) in zip(zip(points, points[1:]),
                                        zip(tangents_y, tangents_y[1:])):
            Cy = np.array([p1[1], p2[1], t1, t2])
            # make it a column vector
            Cy = Cy[:, np.newaxis]
            self.Cys.append(Cy)

    def __call__(self, t, request=None):
        if request is None:
            # for consistency with notes linked above
            s = t
            S = np.array([s ** 3, s ** 2, s, 1])
            # catmull curves are made up of a number of individual curves. Assuming
            # osu! weights each curve equally (that is, each curve takes an equal
            # amount of time to traverse regardless of its size), we can get the
            # curve that should be used for a certain t by multiplying by the
            # number of curves and rounding up.
            if t == 0:
                t += 0.01
            curve_index = math.ceil(t * len(self.Cxs)) - 1
            Cx = self.Cxs[curve_index]
            Cy = self.Cys[curve_index]

            px = (S @ self.h) @ Cx
            py = (S @ self.h) @ Cy
            # A bit of dimensional analysis:
            # S = 1x4
            # C = 4x1
            # h = 4x4
            # P = (S * h) * C = (1x4 * 4x4) * 4x1 = 1x4 * 4x1 = 1x1
            # Result of multiplication is a 1x1 ndarray so convert to a float.
            px = float(px)
            py = float(py)
            return [px, py]
            
        else:
            _, dir = getVector(self(t[0]),self(t[1]))
            return dir


def get_center(a, b, c):
    """Returns the Position of the center of the circle described by the 3 points
    Parameters
    ----------
    a, b, c : Position
        The three positions.
    Returns
    -------
    center : Position
        The center of the three points.
    Notes
    -----
    This uses the same algorithm as osu!
    https://github.com/ppy/osu/blob/7fbbe74b65e7e399072c198604e9db09fb729626/osu.Game/Rulesets/Objects/CircularArcApproximator.cs#L23  # noqa
    """
    a, b, c = np.array([a, b, c], dtype=np.float64)

    a_squared = np.sum(np.square(b - c))
    b_squared = np.sum(np.square(a - c))
    c_squared = np.sum(np.square(a - b))

    if np.isclose([a_squared, b_squared, c_squared], 0).any():
        raise ValueError(f'given points are collinear: {a}, {b}, {c}')

    s = a_squared * (b_squared + c_squared - a_squared)
    t = b_squared * (a_squared + c_squared - b_squared)
    u = c_squared * (a_squared + b_squared - c_squared)

    sum_ = s + t + u

    if np.isclose(sum_, 0):
        raise ValueError(f'given points are collinear: {a}, {b}, {c}')

    return (s * a + t * b + u * c) / sum_


def rotate(position, center, radians):
    """Returns a Position rotated r radians around centre c from p
    Parameters
    ----------
    position : Position
        The position to rotate.
    center : Position
        The point to rotate about.
    radians : float
        The number of radians to rotate ``position`` by.
    """
    p_x, p_y = position
    c_x, c_y = center

    x_dist = p_x - c_x
    y_dist = p_y - c_y

    return [(x_dist * math.cos(radians) - y_dist * math.sin(radians)) + c_x,
            (x_dist * math.sin(radians) + y_dist * math.cos(radians)) + c_y]


def getVector(p1,p2):
    """Returns vector magnitude as well as unit direction"""
    if isinstance(p1,list):
        p1 = np.array(p1).flatten()
    if isinstance(p2,list):
        p2 = np.array(p2).flatten()
    if np.array_equal(p1,p2):
        return 0, np.zeros((len(p1),))
    ovec = p2 - p1
    # magnitude and direction
    mag = np.linalg.norm(ovec)
    return mag, ovec / mag



if __name__ == '__main__':
    # testing Bezier
    test = [[234, 204], [316, 192], [293, 227], [382, 215]]
    length = 148.0
    cv = Bezier(test,length)
    
    # testing Linear
    test = [[189, 244], [199, 244], [219, 244]]
    length = 32.0
    cv = Linear(test,length)
    print(cv(0.2))
    print(cv(0.5))
    print(cv([0.2,0.5],1))
    
    # testing Perfect
    test = [[121,305],[123,276],[121,248]]
    length = 58.1250022172929
    cv = Perfect(test,length)
    
    # testing Catmull
    test = [[234, 204], [316, 192], [293, 227], [382, 215]]
    length = 148.0
    cv = Catmull(test,length)
    print(cv([0.3,0.4],1))
    
    # testing multiBezier
    # long af curve courtesy of Hollow Wings
    test = [[159, 373], [144, 352], [93, 339], [93, 339], 
            [89, 277], [60, 217], [60, 217], [116, 169],
            [144, 106], [144, 106], [199, 118], [268, 98],
            [268, 98], [301, 152], [352, 187], [352, 187],
            [329, 241], [347, 299], [347, 299], [291, 330],
            [269, 370]]
    length = 813.7500310421
    cv = MultiBezier(test,length)
    print(cv([0.1,0.2],1))
    
    # testing Curve class
    # collinear points
    test = [[34, 45], [44, 55], [54, 65]]
    length = 28.45
    cv = Curve.from_kind_and_points('P',test,length)
