import numpy as np
from scipy.spatial.distance import cosine

def extract_cosine_distances(window):
    """
    Wyciąga odległości cosinusowe między parami osi czujników (ac_* i g_*) z ramki czasowej.

    Zwraca:
        dict: cechy odległości
    """
    features = {}

    # Akcelerometr
    features['cos_ac_xy'] = 1 - cosine(window['ac_x'], window['ac_y'])
    features['cos_ac_xz'] = 1 - cosine(window['ac_x'], window['ac_z'])
    features['cos_ac_yz'] = 1 - cosine(window['ac_y'], window['ac_z'])

    # Żyroskop
    features['cos_g_xy'] = 1 - cosine(window['g_x'], window['g_y'])
    features['cos_g_xz'] = 1 - cosine(window['g_x'], window['g_z'])
    features['cos_g_yz'] = 1 - cosine(window['g_y'], window['g_z'])

    return features
