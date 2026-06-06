# Capture_EMG_Data

Acquisition tool for labeled surface EMG (sEMG) data using the **Myo armband** and an ordinary webcam.
The program records pre-defined elbow-angle categories, pairing each sEMG sample window with a
vision-based angle label, and exports the result to CSV for downstream gesture/posture classification.

> Coleta de dados de sEMG rotulados com o bracelete **Myo** e uma webcam comum. O programa grava
> categorias pré-definidas de ângulo de cotovelo, associando cada janela de sinal a um rótulo de ângulo
> obtido por visão computacional, e exporta tudo em CSV para classificação posterior.

---

## Suggested repository metadata

Set these on GitHub (the "About" gear, top-right of the repo page) — they are the single biggest lever for discoverability:

- **Description:** `Labeled sEMG data acquisition from the Myo armband + webcam, with a PyQt5 GUI. Exports elbow-angle categories to CSV.`
- **Topics:** `emg` `semg` `myo-armband` `data-acquisition` `gesture-recognition` `pyqt5` `opencv` `python` `biosignals` `human-machine-interface`

---

## Features

- Real-time sEMG capture from the Myo armband (8 channels).
- Two acquisition modes:
  - **Filtered, ~50 Hz** — `capture_myo_filtered_signal_50hz.py`
  - **Raw, ~200 Hz** — `capture_myo_not_filtered_signal_200hz.py`
- PyQt5 graphical interface for configuring and running a session.
- Webcam-based elbow-angle estimation used to label each capture.
- Configurable number of categories (2–4) with an adjustable angle tolerance per category.
- Automatic CSV export at the end of a session.

## How it works

Through the GUI you:

1. Select the arm being measured.
2. Set the number of samples per category (the Myo streams ~200 samples/s).
3. Choose the number of categories (2–4) and the angle variance accepted for a sample to belong to a category.
   Avoid overlapping ranges — they cause label noise.
4. Press **Start**.

While an angle is being captured, the on-screen arm overlay turns **orange**; once a category reaches its
target sample count, it turns **green**. When all categories are filled, the capture window closes and the
data is saved automatically.

Default angle-to-category mapping:

| Categories | Mapping |
|------------|---------|
| 2 | 170° → cat 1, 90° → cat 2 |
| 3 | 170° → cat 1, 90° → cat 2, 60° → cat 3 |
| 4 | 170° → cat 1, 90° → cat 2, 60° → cat 3, 45° → cat 4 |

## Requirements

- Tested on **Linux (Ubuntu 20.04)**.
- Myo armband with its original Bluetooth USB adapter connected to a USB port.
- Python dependencies are listed in [`requirements.txt`](requirements.txt).

Install:

```bash
pip3 install -r requirements.txt
```

### Troubleshooting

- **PyQt5 / `xcb` platform plugin error** (often an OpenCV ↔ PyQt5 conflict):
  ```bash
  pip3 install PyQt5==5.11.3
  ```
  Reference: https://stackoverflow.com/questions/63829991/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it
- **`cv2` import problems:**
  ```bash
  pip install opencv-contrib-python  # works on both Windows and Ubuntu
  ```
  Reference: https://stackoverflow.com/questions/37776228/pycharm-python-opencv-and-cv2-install-error

## Usage

```bash
# filtered signal at ~50 Hz
python3 capture_myo_filtered_signal_50hz.py

# raw signal at ~200 Hz
python3 capture_myo_not_filtered_signal_200hz.py
```

Captured datasets are written to the `EMG_Data/` directory as CSV files.

## Repository layout

| Path | Purpose |
|------|---------|
| `capture_myo_filtered_signal_50hz.py` | Capture session, filtered signal (~50 Hz) |
| `capture_myo_not_filtered_signal_200hz.py` | Capture session, raw signal (~200 Hz) |
| `pose_module.py` | Webcam-based pose/angle estimation |
| `common.py` | Shared helpers |
| `ui_main_window.py`, `ui_main_window.ui` | PyQt5 interface |
| `turn_off.py` | Utility to power down / disconnect the Myo |
| `EMG_Data/` | Output datasets |
| `libmyolinux/`, `myo-python/`, `myo-raw/` | Myo communication backends |

## Acknowledgments

This project builds on the work of several people who made Myo communication on Linux possible:

- **dzhu** — `myo-raw`, the base library for communicating with the Myo: https://github.com/dzhu/myo-raw
- **Alvipe** — additional improvements to `myo-raw`, incorporated here: https://github.com/Alvipe/myo-raw
- **Fernando Cossentino** — PyoConnect, modifications used in this project: http://www.fernandocosentino.net/pyoconnect/ (accessed 2022-06-20)
- **Alan Mendes (alans96)** — base version of the project without the EMG-reading module: https://github.com/alans96/arm_robotics

## License

Released under the **Apache-2.0 License**. See [`LICENSE`](LICENSE).

## Author

**Caio Lima** — UFABC
[GitHub](https://github.com/Calima94) · [LinkedIn](https://www.linkedin.com/in/caio-lima-9035a022a/) · [ResearchGate](https://www.researchgate.net/profile/Caio-Lima-15) · [Lattes](http://lattes.cnpq.br/0127370029893676)
