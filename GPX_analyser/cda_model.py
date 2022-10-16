import pandas as pd
import numpy as np

def cda(mass, gravity, velocity_rider, r_r, s, watts, rho, acceleration, velocity_air):
    """
    https://blog.flocycling.com/aero-wheels/flo-cycling-studying-tires-part-7-solving-for-cda-1/

    """
    tire_rr = r_r*mass*gravity*velocity_rider
    slope_r = s*mass*gravity
    accel = mass*acceleration*velocity_rider
    air = rho*(velocity_air**2)*velocity_rider
    CdA = 2*(w - tire_rr - slope_r - accel) / air

    return CdA