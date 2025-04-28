import numpy as np

def calculate_accelerometer_magnitude(window):
    """
    Oblicza średnią wartość wektora wielkości (sqrt(x^2 + y^2 + z^2)) dla akcelerometru w oknie czasowym

    Argumenty:
        window: DataFrame reprezentujący pojedyncze okno czasowe

    Zwraca:
        float: Średnia wartość wektora wielkości dla akcelerometru
    """
    axes = ['ac_x', 'ac_y', 'ac_z']

    for axis in axes:
        if axis not in window.columns:
            raise ValueError(f"Kolumna '{axis}' nie została znaleziona w DataFrame.")

    vector_magnitude = np.sqrt(window['ac_x']**2 + window['ac_y']**2 + window['ac_z']**2)
    return vector_magnitude.mean()

def calculate_gyroscope_magnitude(window):
    """
    Oblicza średnią wartość wektora wielkości (sqrt(x^2 + y^2 + z^2)) dla żyroskopu w oknie czasowym

    Argumenty:
        window: DataFrame reprezentujący pojedyncze okno czasowe

    Zwraca:
        float: Średnia wartość wektora wielkości dla żyroskopu
    """
    axes = ['g_x', 'g_y', 'g_z']

    for axis in axes:
        if axis not in window.columns:
            raise ValueError(f"Kolumna '{axis}' nie została znaleziona w DataFrame.")

    vector_magnitude = np.sqrt(window['g_x']**2 + window['g_y']**2 + window['g_z']**2)
    return vector_magnitude.mean()