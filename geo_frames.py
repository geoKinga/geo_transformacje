# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:36:33 2022

@author: dell
"""

from math import pi, sin, cos, acos, tan, sqrt, atan, atan2, degrees, radians
import numpy as np

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 z obliczenia 0.0818191908426201

        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2

    def deg2dms(self, decimal_deg):
        '''
        This method convert decimal degree to deg ,min, sec
        '''
        d = int(decimal_deg)
        m = int((decimal_deg - d) * 60)
        s = (decimal_deg - d - m/60.) * 3600.
        return(d, m, s)
    
    def dms2deg(self, dms):
        '''
        This method convert deg ,min, sec to decimal degree to 
        '''
        d = dms[0]; m = dms[1]; s = dms[2]
        decimal_degree = d + m/60 + s/3600
        return(decimal_degree)
    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (x, y, z)
        na współrzędne geodezyjne długość szerokość i wysokośc elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
        Parameters
        ----------
        X, Y, Z : FLOAT
             współrzędne w układzie orto-kartezjańskim, 

        Returns
        -------
        lat
            [stopnie dziesiętne] - szerokość geodezyjna
        lon
            [stopnie dziesiętne] - długośc geodezyjna.
        h : TYPE
            [metry] - wysokość elipsoidalna
        output [STR] - optional, defoulf 
            dec_degree - decimal degree
            dms - degree, minutes, sec
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            
    def xyz2plh2(self, x, y, z):
        eps = self.ecc2 / (1.0 - self.ecc2)
        r = sqrt(x * x + y * y) # p
        q = atan2((z * self.a), (r * self.b)) #q - aprox phi 
        phi = atan2((z + eps * self.b * sin(q)**3), (r - self.ecc2 * self.a * cos(q)**3))
        lam = atan2(y, x)      
        N = self.a / sqrt(1.0 - self.ecc2 * sin(phi) **2)
        h   = (r / cos(phi)) - N
        return(degrees(phi),degrees(lam),h)
    
    def plh2xyz(self, lat, lon, h):
        '''
        This method returns the position coordinate [X,Y,Z] given in the Earth Centered Earth Fixed 
        (ECEF) coordinate  for a user located at the goedetic coordinates lat,lon and h. The units 
        of the output position vector, p_e, are meters. latitude, longitude, altitude (reference 
        elipsoid)
        INPUT:

            lat : latitude [degree] 
            lon : longitude [degree] 
            h   : altitude [meter]
            
        OUTPUT: 
            X,Y,Z  - geocentric coordinates
        '''
        #   Compute East-West Radius of curvature at current position
        R_E = self.a/(sqrt(1. - (self.ecc * sin(radians(lat)))**2))
        #  Compute ECEF coordinates
        X = (R_E + h)*cos(radians(lat)) * cos(radians(lon))
        Y = (R_E + h)*cos(radians(lat)) * sin(radians(lon))
        Z = ((1 - self.ecc**2)*R_E + h)*sin(radians(lat))
        return(X, Y, Z)

    def ll2gk(self, lat, lon, L0):
        """
        latitude longitude WGS 84 to xy gaus-krugger
        Czarnecki pp. 60-62
        ll2gk(52.0, 21.0, 21) = 5763343.550362656, 0.0
        """
        t = tan(lat) 
        b2 = self.a**2 * (1-self.ecc2) 
        ep2 = (self.a**2 - b2)/b2 
        n2 = ep2 * (cos(lat))**2
        l = lon - L0
        # długośc łuku południka???
        A0 = 1 - (self.ecc2/4) - (3*self.ecc2**2/64) - (5*self.ecc2**3/256)
        A2 = (3/8) * (self.ecc2 + (self.ecc2**2/4) + (15*self.ecc2**3/128))
        A4 = (15/256) * (self.ecc2**2 + (3*self.ecc2**3/4))
        A6 = 35*self.ecc2**3/3072
        X = self.a * (A0 * lat - A2 * sin(2 * lat) + A4 * sin(4 * lat) - A6 * sin(6 * lat))
        #
        N = self.a/(1 - self.ecc2 * (sin(lat))**2)**(0.5)
        # xy gauss-kruger
        xgk_h = 1 + (l**2 * (cos(lat))**2 * (5 - t**2 + 9*n2 + 4*n2**2)/12) + (l**4*(cos(lat))**4*(61-58*t**2+t**4+270*n2-330*n2*t**2)/360)
        xgk = X + ((l**2 * N * sin(lat) * cos(lat)/2))* xgk_h
        
        ygk_h = 1 + l**2*(cos(lat))**2*(1-t**2+n2)/6+l**4*(cos(lat))**4*(5-18*t**2+t**4+14*n2-58*n2*t**2)/120
        ygk = (l * N * cos(lat)) * ygk_h
        return xgk, ygk

    def philam2xy2000(self, lat, lon, L0=21):
        """
        latitude longitude WGS 84 to xy 2000
        ll2xy2000(52.00, 21.00, L0=21) > x=5762899.773 y=7500000.000
        """
        xgk, ygk = self.ll2gk(radians(lat), radians(lon), radians(L0))

        x2000 = xgk * 0.999923
        y2000 = ygk * 0.999923 + (L0/3)*1000000+500000
        return x2000, y2000

        
    def philam2xy1992(self, lat, lon, L0=19):
        """
        latitude longitude WGS 84 to xy 2000
        ll2xy1992(52.00, 21.00, L0=19) >  x=461197.243 y=637253.161
        """
        xgk, ygk = self.ll2gk(radians(lat), radians(lon), radians(L0))

        x1992 = xgk * 0.9993 - 5300000
        y1992 = ygk * 0.9993 + 500000
        return x1992 , y1992 
        


    def topocentrENU(self, X0, Y0, Z0, X, Y, Z):
        '''
        This function returns the position coordinates of a user at the WGS-84 ECEF coordiantes in east-north-up coordinates
        relative to the reference position located at lat_ref (latitude in degrees),     
        lon_ref (longitude in degrees) and h_ref (in meters).
        
        INPUT:
            Xuser, Xuser, Xuser : WGS-84 ECEF [meters]
            lat_ref             : [degrees] latitude,     
            lon_ref             : [degrees] longitude, 
            h_ref               : [meters] altitude
        OUTPUT: 
            lat,lon             : [degree], 
            U[ in [meters]
        EXAMPLE: 
            enu = xyz2enu(p_e,lat_ref,lon_ref,h_ref)           
        '''
        #  convert lat, lon, h  to XYZ WGS84 
        delta_X = X - X0 
        delta_Y = Y - Y0 
        delta_Z = Z - Z0
        
        lat_ref, lon_ref, h_ref = self.xyz2plh(X0, Y0, Z0)
        
        #  Calculate ENU coordinates
        East    = -sin(radians(lon_ref)) * delta_X + cos(radians(lon_ref)) * delta_Y
        North   = -sin(radians(lat_ref)) * cos(radians(lon_ref)) * delta_X - sin(radians(lat_ref)) * sin(radians(lon_ref)) * delta_Y + cos(radians(lat_ref)) * delta_Z
        Up      =  cos(radians(lat_ref)) * cos(radians(lon_ref)) * delta_X + cos(radians(lat_ref)) * sin(radians(lon_ref)) * delta_Y + sin(radians(lat_ref)) * delta_Z
        return(East, North, Up)


    def az_el(self, X0, Y0, Z0, X, Y, Z):
        '''
        Transformation of vector dx(sat-rec) into topocentric coordinate system with:
        X0, Y0, Z0  - coordinate ot the onother point
        X, Y, Z     - 
        This modul calculates the azimutr, elevation angle, zenith angle and distance 3D from a reference position specified in ECEF coordinates 
        (e.g. receiver) to another position specified in ECEF coordinates (e.g. satellite ):
        INPUT :
            X0, Y0, Z0 - coordinates of the cener point 
            Xsat, Ysat, Zsat - ECEF satellite coordinates
        OUTPUT: 
            az   - azimuth from north positive clockwise [degrees]
            el   - elevation angle, [degrees]
            ze   - zenit angle
            D    - vector length in units like the input
           
        '''
        dtr = pi/180
        # Transformation of vector dx into topocentric coordinate system with origin at X.    
        [phi, lam, h] = self.xyz2plh(X0, Y0, Z0) # lat, lon, h 
        phi_rad = (phi)*dtr
        lam_rad = (lam)*dtr
        
        #!!! ...
        N = np.array([  [-sin(phi_rad)*cos(lam_rad) ],
                        [-sin(phi_rad)*sin(lam_rad) ],
                        [      cos(phi_rad)         ]])    
        
        E  = np.array([ [-sin(lam_rad)],
                        [ cos(lam_rad)],
                        [       0     ]])
        
        U  = np.array([ [ cos(phi_rad)*cos(lam_rad)],
                        [ cos(phi_rad)*sin(lam_rad)],
                        [      sin(phi_rad)        ]])
        # control: E == N X U        
        # define rotation matrix F
        F  = np.array([[-sin(lam_rad), -sin(phi_rad)*cos(lam_rad), cos(phi_rad)*cos(lam_rad)],
                        [ cos(lam_rad), -sin(phi_rad)*sin(lam_rad), cos(phi_rad)*sin(lam_rad)],
                        [       0,               cos(phi_rad),                 sin(phi_rad)  ]])
        
        #  unit vector from origin of topocentric frame to the point
        dx    = np.array([[X - X0, Y - Y0, Z - Z0]]).T  # 2dim array
       
        local_vector = F.T @ dx
        E = local_vector[0]
        N = local_vector[1]
        U = local_vector[2]
        print(f'ENU: {E[0]:.6f}, {N[0]:.6f}, {U[0]:.6f}')
        
        HZdist = sqrt(E**2 + N**2)
        dist   =  sqrt((X - X0)**2 + (Y - Y0)**2 + (Z - Z0)**2)
        if HZdist < 1.e-20:
            print('distant is too short, probably the same points is used')
            print('Az = 0; El = 0; Z = 0')
            az = 0; El = 0; Z = 0
        else:
            az = (atan2(E,N))/dtr
            ze = (acos(U/dist)) /dtr    #zenith angle
            el = (atan2(U,HZdist))/dtr  #elevation angle
        if az < 0:
            az = az + 360
        d = sqrt(dx[0,0]**2 + dx[1,0]**2 + dx[2,0]**2);
        return az, el, ze, d    # [decimal_degree, meters]



if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    # --- xyz2plh
    phi, lam, h = geo.xyz2plh(X, Y, Z)
    print(f"plh: {phi:.7f}, {lam:.7f}, {h:.3f}")
    #phi, lam, h = geo.xyz2plh2(X, Y, Z)
    #print(f"{phi:.7f}, {lam:..7f}, {h:.3f}")
    plh = geo.xyz2plh(X, Y, Z, output = 'dms')
    #print(plh)
    # --- plh2xyz
    x, y, z = geo.plh2xyz(phi, lam, h)
    print(f"xyz :{x:.3f}, {y:.3f}, {z:.3f}")
    # --- [plh2xy200]
    x2000, y2000 = geo.philam2xy2000(phi, lam, L0=21)
    print(f"xy2000: {x2000:.3f}, {y2000:.3f}")
    # --- [plh2xy200]
    x1992, y1992 = geo.philam2xy1992(phi, lam)
    print(f"xy1992: {x1992:.3f}, {y1992:.3f}")
    # --- [topocentric]
    E, N, U =  geo.topocentrENU(X, Y, Z, X+1, Y+1, Z+1)
    print(f'ENU: {E:.6f}, {N:.6f}, {U:.6f}')
    # --- [az_el]
    az, el, ze, d =  geo.az_el(X, Y, Z, X+1, Y+1, Z+1)
    full = el + ze
    print(f'az_el: {az:.6f}, {el:.6f}, {ze:.6f}, {ze:.3f}, control = {full}')

    
        
    
