#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from geo_frames import *
geo = Transformacje(model = "wgs84")
#import sys
#sys.path.append('C:\\Users\\kinga\\projekt')



plik = "wsp_inp.txt"
# odczyt z pliku: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
tablica = np.genfromtxt(plik, delimiter=',', skip_header = 4)

## tutaj pętla i przeliczenie do wszystkich układów .....




# zapis: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.savetxt.html
np.savetxt("wsp_out.txt", tablica, delimiter=',', fmt = ['%10.2f', '%10.2f', '%10.3f'], header = 'konwersja współrzednych geodezyjnych \\ kinga węzka')
