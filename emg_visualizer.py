"""
Real-time 8-channel EMG visualizer for MindRove armband.

Usage:
    python emg_visualizer.py              # live from armband
    python emg_visualizer.py --synthetic  # fake data for testing without hardware

Press Ctrl+C or close the window to quit.
"""

import argparse
import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 5          # how many seconds of history to show
UPDATE_MS = 33              # ~30 fps refresh
BANDPASS_LOW = 20.0         # Hz — below this is motion artifact
BANDPASS_HIGH = 450.0       # Hz — above this is noise (Nyquist ≈ 250)
NOTCH_FREQ = 60.0           # power line interference (60 Hz US, change to 50 for EU)
FILTER_ORDER = 4

# Channel colors (8 distinct)
COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45']

class EMGVisualizer(QtWidgets.QMainWindow):
    def __init__(self, board: BoardShim, board_id: int):
        super().__init__()
        self.board = board
        self.board_id = board_id
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.emg_channels = BoardShim.get_emg_channels(board_id)[:8]  # cap at 8 for display
        self.accel_channels = BoardShim.get_accel_channels(board_id)
        self.gyro_channels = BoardShim.get_gyro_channels(board_id)
        self.num_emg = len(self.emg_channels)
        self.buf_size = int(WINDOW_SECONDS * self.sampling_rate)

        # Ring buffers for display
        self.emg_buf = np.zeros((self.num_emg, self.buf_size))
        self.accel_buf = np.zeros((3, self.buf_size))
        self.gyro_buf = np.zeros((3, self.buf_size))
        self.time_axis = np.linspace(-WINDOW_SECONDS, 0, self.buf_size)

        self._init_ui()
        self._start_timer()

    def _init_ui(self):
        self.setWindowTitle('MindRove EMG — Real-time')
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Graphics layout for all plots
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground('#1e1e1e')
        layout.addWidget(self.graphics)

        # --- 8 EMG channels (stacked) ---
        self.emg_plots = []
        self.emg_curves = []
        for i in range(self.num_emg):
            if i > 0:
                self.graphics.nextRow()
            p = self.graphics.addPlot(title=f'EMG {i+1}')
            p.setLabel('left', 'µV')
            p.showGrid(y=True, alpha=0.3)
            p.setXRange(-WINDOW_SECONDS, 0, padding=0)
            p.setYRange(-500, 500)
            p.hideAxis('bottom') if i < self.num_emg - 1 else p.setLabel('bottom', 's')
            # Link x axes
            if i > 0:
                p.setXLink(self.emg_plots[0])
            curve = p.plot(pen=pg.mkPen(COLORS[i], width=1.5))
            self.emg_plots.append(p)
            self.emg_curves.append(curve)

        # --- IMU row (accel + gyro side by side) ---
        self.graphics.nextRow()
        self.accel_plot = self.graphics.addPlot(title='Accelerometer')
        self.accel_plot.setLabel('left', 'g')
        self.accel_plot.setLabel('bottom', 's')
        self.accel_plot.addLegend(offset=(60, 10))
        self.accel_plot.showGrid(y=True, alpha=0.3)
        self.accel_plot.setXRange(-WINDOW_SECONDS, 0, padding=0)
        self.accel_curves = []
        for i, axis in enumerate(['X', 'Y', 'Z']):
            c = self.accel_plot.plot(pen=pg.mkPen(COLORS[i], width=1.5), name=axis)
            self.accel_curves.append(c)

        self.gyro_plot = self.graphics.addPlot(title='Gyroscope')
        self.gyro_plot.setLabel('left', '°/s')
        self.gyro_plot.setLabel('bottom', 's')
        self.gyro_plot.addLegend(offset=(60, 10))
        self.gyro_plot.showGrid(y=True, alpha=0.3)
        self.gyro_plot.setXRange(-WINDOW_SECONDS, 0, padding=0)
        self.gyro_curves = []
        for i, axis in enumerate(['X', 'Y', 'Z']):
            c = self.gyro_plot.plot(pen=pg.mkPen(COLORS[i + 3], width=1.5), name=axis)
            self.gyro_curves.append(c)

    def _start_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(UPDATE_MS)

    def _update(self):
        data = self.board.get_board_data()  # pop all new samples
        if data.shape[1] == 0:
            return

        # --- EMG ---
        for i, ch in enumerate(self.emg_channels):
            raw = data[ch]
            # Filter: detrend → bandpass → notch
            DataFilter.detrend(raw, DetrendOperations.LINEAR.value)
            DataFilter.perform_bandpass(
                raw, self.sampling_rate, BANDPASS_LOW, BANDPASS_HIGH,
                FILTER_ORDER, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.remove_environmental_noise(raw, self.sampling_rate, 1)  # 60 Hz
            # Append to ring buffer
            n = len(raw)
            self.emg_buf[i] = np.roll(self.emg_buf[i], -n)
            self.emg_buf[i, -n:] = raw
            self.emg_curves[i].setData(self.time_axis, self.emg_buf[i])

        # --- Accel ---
        for i, ch in enumerate(self.accel_channels):
            n = data.shape[1]
            self.accel_buf[i] = np.roll(self.accel_buf[i], -n)
            self.accel_buf[i, -n:] = data[ch]
            self.accel_curves[i].setData(self.time_axis, self.accel_buf[i])

        # --- Gyro ---
        for i, ch in enumerate(self.gyro_channels):
            n = data.shape[1]
            self.gyro_buf[i] = np.roll(self.gyro_buf[i], -n)
            self.gyro_buf[i, -n:] = data[ch]
            self.gyro_curves[i].setData(self.time_axis, self.gyro_buf[i])


def main():
    parser = argparse.ArgumentParser(description='MindRove EMG real-time visualizer')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic board (no hardware)')
    parser.add_argument('--ip', default='192.168.4.1', help='Armband IP address')
    parser.add_argument('--port', type=int, default=4210, help='Armband port')
    args = parser.parse_args()

    # --- Board setup ---
    BoardShim.enable_board_logger()

    if args.synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD
        params = MindRoveInputParams()
    else:
        board_id = BoardIds.MINDROVE_WIFI_BOARD
        params = MindRoveInputParams()
        params.ip_address = args.ip
        params.ip_port = args.port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    print(f'Streaming from {"SYNTHETIC" if args.synthetic else "MindRove armband"} '
          f'@ {BoardShim.get_sampling_rate(board_id)} Hz')
    print(f'EMG channels: {BoardShim.get_emg_channels(board_id)}')

    # --- Qt app ---
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = EMGVisualizer(board, board_id)
    window.show()

    try:
        app.exec_()
    finally:
        board.stop_stream()
        board.release_session()
        print('Session released.')


if __name__ == '__main__':
    main()
