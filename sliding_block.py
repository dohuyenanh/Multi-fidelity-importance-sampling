"""Model a block stopping under constant kinetic friction on a horizontal floor.

The work-energy model supplies the reference stopping distance. Its tangent
at 9 m/s supplies the surrogate. Both take the same initial speed in m/s and
use a line 24.01 m from the starting point. The experiment draws nominal
speeds uniformly from 0 to 10 m/s. These functions do not sample speeds."""
import numpy as np

FRICTION = 0.20
GRAVITY = 10.0
DECELERATION = FRICTION * GRAVITY
LINE_DISTANCE = 24.01
TANGENT_SPEED = 9.0
SPEED_BOUNDS = (0.0, 10.0)


def stopping_distance(speed):
    """Return the work-energy stopping distance in metres.

    speed is a nonnegative initial speed in m/s, supplied as a scalar or
    array-like object. Friction removes kinetic energy at a constant rate
    per metre, giving speed squared divided by twice the deceleration.
    Array inputs produce an array of distances with the same shape."""
    return np.asarray(speed)**2 / (2 * DECELERATION)


def tangent_distance(speed):
    """Approximate stopping distance in metres by the tangent at TANGENT_SPEED.

    speed is a scalar or array-like initial speed in m/s. The tangent matches
    the reference distance and its slope at 9 m/s. It underestimates the convex
    distance curve elsewhere and can give negative values far from the point
    of tangency. Outputs follow the input shape and are not clipped."""
    slope = TANGENT_SPEED / DECELERATION
    distance_at_tangent = stopping_distance(TANGENT_SPEED)
    speed_change = np.asarray(speed) - TANGENT_SPEED
    return distance_at_tangent + slope * speed_change


def crosses_line(speed):
    """Classify a crossing using the work-energy stopping distance.

    Pass one initial speed in m/s or an array of speeds. Return a Boolean
    per speed, true only if the block travels strictly beyond LINE_DISTANCE.
    Stopping exactly on the line counts as a non-crossing."""
    # Comparing speeds avoids rounding a stop on the line into a crossing.
    speed_at_line = np.sqrt(2 * DECELERATION * LINE_DISTANCE)
    return np.asarray(speed) > speed_at_line


def surrogate_crosses_line(speed):
    """Classify a crossing using the tangent approximation.

    Accept the same scalar or array-like speeds in m/s as crosses_line and
    return Boolean values of the same shape. A predicted distance strictly
    above LINE_DISTANCE counts as a crossing. MFIS uses these labels to
    fit its proposal during the pilot stage."""
    return tangent_distance(speed) > LINE_DISTANCE
