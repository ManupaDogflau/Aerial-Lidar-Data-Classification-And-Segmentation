# geometry.py
import math

def polar_to_xyz(dist, az_deg, vert_deg):
    az = math.radians(az_deg)
    vert = math.radians(vert_deg)
    x = dist * math.cos(vert) * math.cos(az)
    y = dist * math.cos(vert) * math.sin(az)
    z = dist * math.sin(vert)
    return x, y, z
