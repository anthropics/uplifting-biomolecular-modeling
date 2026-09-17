import os


def get_information(length):
    if length < 60:
        return {
            "AF2_ipae": [3.84, 5.03, 6.07, 6.12, 21.36, 10.57, 8.75, 8.24, 23.1, 15.32],
            "ptx_iptm": [
                0.95,
                0.9,
                0.83,
                0.64,
                0.54,
                0.49,
                0.89,
                0.67,
                0.28,
                0.1,
            ],
            "ptx_temp_iptm": [
                0.95,
                0.91,
                0.87,
                0.84,
                0.44,
                0.6,
                0.88,
                0.58,
                0.68,
                0.35,
            ],
            "AF2_SR": [44.89, 37.5, 28.41, 19.6, 3.12, 10.23, 26.42, 13.92, 0.28, 5.68],
            "ptx_sr": [55.7, 32.4, 11.9, 3.1, 2.3, 0.3, 33.0, 3.7, 0.0, 0.0],
            "ptx_temp_sr": [58.5, 50.0, 17.0, 15.9, 0.6, 0.0, 31.2, 0.6, 1.1, 0.0],
            "count": "Length 50-60",
        }
    elif length < 80 and length >= 60:
        return {
            "AF2_ipae": [3.99, 4.97, 5.73, 6.37, 17.19, 9.37, 8.83, 8.04, 22.71, 16.09],
            "ptx_iptm": [
                0.94,
                0.9,
                0.88,
                0.59,
                0.58,
                0.52,
                0.88,
                0.76,
                0.33,
                0.09,
            ],
            "ptx_temp_iptm": [
                0.95,
                0.91,
                0.88,
                0.86,
                0.54,
                0.58,
                0.88,
                0.66,
                0.71,
                0.36,
            ],
            "AF2_SR": [
                44.79,
                44.35,
                33.93,
                20.39,
                5.65,
                13.39,
                28.87,
                19.49,
                0.74,
                6.55,
            ],
            "ptx_sr": [54.6, 36.6, 16.5, 2.8, 2.2, 0.1, 33.5, 7.1, 0.0, 0.0],
            "ptx_temp_sr": [62.5, 45.2, 22.0, 19.6, 0.9, 0.0, 37.5, 3.9, 3.9, 0.0],
            "count": "Length 60-80",
        }
    elif length < 100 and length >= 80:
        return {
            "AF2_ipae": [4.03, 5.08, 5.58, 6.12, 14.36, 8.87, 9.79, 7.96, 22.57, 16.02],
            "ptx_iptm": [
                0.94,
                0.88,
                0.86,
                0.56,
                0.64,
                0.51,
                0.87,
                0.82,
                0.33,
                0.09,
            ],
            "ptx_temp_iptm": [
                0.96,
                0.9,
                0.88,
                0.87,
                0.61,
                0.59,
                0.88,
                0.71,
                0.72,
                0.37,
            ],
            "AF2_SR": [49.4, 45.09, 38.24, 21.28, 6.55, 15.18, 16.22, 23.51, 0.6, 4.91],
            "ptx_sr": [59.4, 30.2, 18.9, 2.1, 3.6, 0.0, 24.4, 9.1, 0.0, 0.0],
            "ptx_temp_sr": [70.5, 43.5, 23.2, 26.5, 1.8, 0.0, 29.2, 6.2, 5.4, 0.0],
            "count": "Length 80-100",
        }
    elif length < 120 and length >= 100:
        return {
            "AF2_ipae": [
                4.08,
                5.28,
                5.69,
                6.73,
                12.75,
                8.38,
                10.29,
                8.06,
                22.85,
                13.73,
            ],
            "ptx_iptm": [
                0.94,
                0.88,
                0.84,
                0.44,
                0.6,
                0.54,
                0.85,
                0.84,
                0.26,
                0.09,
            ],
            "ptx_temp_iptm": [
                0.96,
                0.9,
                0.87,
                0.88,
                0.55,
                0.58,
                0.86,
                0.79,
                0.69,
                0.47,
            ],
            "AF2_SR": [47.77, 38.99, 42.11, 15.62, 7.59, 16.52, 12.5, 25.3, 1.34, 5.95],
            "ptx_sr": [60.3, 25.1, 15.6, 1.3, 2.5, 0.0, 18.9, 14.3, 0.0, 0.0],
            "ptx_temp_sr": [71.4, 33.9, 20.5, 23.2, 1.5, 0.0, 21.7, 9.5, 4.2, 0.0],
            "count": "Length 100-120",
        }
    elif length < 140 and length >= 120:
        return {
            "AF2_ipae": [4.16, 5.51, 5.78, 7.66, 8.48, 9.19, 11.08, 8.12, 22.62, 14.74],
            "ptx_iptm": [
                0.93,
                0.87,
                0.83,
                0.37,
                0.62,
                0.51,
                0.83,
                0.84,
                0.27,
                0.09,
            ],
            "ptx_temp_iptm": [
                0.95,
                0.87,
                0.87,
                0.88,
                0.65,
                0.59,
                0.82,
                0.85,
                0.74,
                0.52,
            ],
            "AF2_SR": [48.07, 34.82, 43.45, 12.5, 13.1, 12.8, 8.04, 22.47, 1.19, 4.61],
            "ptx_sr": [52.2, 22.6, 11.6, 0.6, 1.6, 0.3, 14.4, 15.2, 0.0, 0.0],
            "ptx_temp_sr": [66.7, 23.2, 21.4, 19.0, 4.2, 0.0, 14.6, 11.9, 4.2, 0.0],
            "count": "Length 120-140",
        }
    else:
        return {
            "AF2_ipae": [4.22, 5.56, 5.74, 9.47, 6.74, 9.74, 11.38, 8.37, 23.94, 17.1],
            "ptx_iptm": [
                0.93,
                0.86,
                0.8,
                0.31,
                0.72,
                0.52,
                0.81,
                0.83,
                0.23,
                0.09,
            ],
            "ptx_temp_iptm": [
                0.95,
                0.87,
                0.89,
                0.84,
                0.71,
                0.56,
                0.79,
                0.87,
                0.72,
                0.42,
            ],
            "AF2_SR": [45.17, 32.95, 39.2, 10.51, 22.73, 11.36, 7.95, 19.6, 1.7, 4.26],
            "ptx_sr": [34.7, 18.8, 9.7, 0.6, 3.4, 0.3, 8.2, 22.4, 0.0, 0.0],
            "ptx_temp_sr": [57.4, 21.0, 23.9, 11.9, 6.2, 0.0, 8.5, 19.9, 4.5, 0.0],
            "count": "Length 140-150",
        }


labels = [
    "BHRF1",
    "PDL1",
    "IR",
    "TrkA",
    "SC2RBD",
    "VEGF-A",
    "IL7RA",
    "H1",
    "IL17A",
    "TNFa",
]

difficulty_to_color = {
    "easy": "#086BFFE8",
    "hard": "#086BFFE8",
    "very_hard": "#086BFFE8",
}
font_path = os.path.join(os.path.dirname(__file__), "Helvetica-Regular.ttf")

text_full = (
    "Your submitted case is positioned at the difficulty level shown below, based on evaluations from"
    + " both AlphaFold2 model (initial guess) and Protenix model. If you believe the target you provided does not"
    + " match the expected difficulty, you may adjust the input model settings: hotspot location, binder length,"
    + " and target file resolution."
)

text_preview = (
    "Your submitted case is positioned at the difficulty level shown below, based on evaluations from"
    + " AlphaFold2 model (initial guess). If you believe the target you provided does not"
    + " match the expected difficulty, you may adjust the input model settings: hotspot location, binder length,"
    + " and target file resolution."
)
