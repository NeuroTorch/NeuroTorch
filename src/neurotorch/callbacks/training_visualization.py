import os.path
import pickle
import time
import multiprocessing as mp
from typing import Union, Optional

from matplotlib import pyplot as plt

from .base_callback import BaseCallback


class _VisualizerProcess(mp.Process):
    """
    Process to load a training history from pickle file and plot it. The process will update the plot every `update_dt`
    seconds and will close when join method is called.

    :Attributes:
        - **fig** (plt.Figure): The figure to plot.
        - **axes** (plt.Axes): The axes to plot.
        - **lines** (list): The list of lines to plot.
    """
    def __init__(self, history_path: str, lock: Union[mp.Lock, mp.RLock], update_dt: float = 1.0):
        """
        Create a new process to plot the training history.

        :param history_path: The path to the pickle file containing the training history.
        :type history_path: str
        :param lock: The lock to use to protect the training history.
        :type lock: Union[mp.Lock, mp.RLock]
        :param update_dt: The time in seconds between two updates of the plot.
        :type update_dt: float
        """
        super().__init__()
        self._history_path = history_path
        self._lock = lock
        self._update_dt = update_dt

        self._close_event = mp.Event()
        self._training_history = None
        self.fig, self.axes, self.lines = None, None, None

    def run(self):
        """
        Run the process.

        :return: None
        """
        can_plot = False
        while not can_plot and not self._close_event.is_set():
            if os.path.exists(self._history_path):
                try:
                    with self._lock:
                        self._training_history = pickle.load(open(self._history_path, "rb"))
                    can_plot = True
                except Exception:
                    time.sleep(self._update_dt)

        while len(self._training_history) < 2 and not self._close_event.is_set():
            self.update_history()
            time.sleep(self._update_dt)

        plt.ion()
        self.fig, self.axes, self.lines = self._training_history.create_plot()
        plt.pause(self._update_dt)
        while not self._close_event.is_set():
            self.update_history()
            self.fig, self.axes, self.lines = self._training_history.update_fig(self.fig, self.axes, self.lines)
            plt.pause(self._update_dt)
        plt.close(self.fig)

    def update_history(self):
        """
        Update the training history by loading it from the pickle file.

        :return: None
        """
        try:
            with self._lock:
                self._training_history = pickle.load(open(self._history_path, "rb"))
        except Exception as e:
            print(f"{e = }")
            pass

    def join(self, *args, **kwargs):
        """
        Set the close event.
        Join the current process.
        """
        self._close_event.set()
        plt.close(self.fig)
        return super().join(*args, **kwargs)


class TrainingHistoryVisualizationCallback(BaseCallback):
    """
    This callback is used to visualize the training history in real time.

    :Attributes:
        - **fig** (plt.Figure): The figure to plot.
        - **axes** (plt.Axes): The axes to plot.
        - **lines** (list): The list of lines to plot.
    """

    def __init__(
            self,
            temp_folder: str = '~/temp/',
            **kwargs
    ):
        """
        Create a new callback to visualize the training history.

        :param temp_folder: The folder where to save the training history.
        :type temp_folder: str
        :param kwargs: The keyword arguments to pass to the base callback.
        """
        super().__init__(**kwargs)
        self._is_open = False
        os.makedirs(temp_folder, exist_ok=True)
        self._temp_path = os.path.join(
            os.path.expanduser(temp_folder), f'training_visualization{time.time_ns()}.pkl'
        )
        self._lock = mp.Lock()
        self._process = _VisualizerProcess(self._temp_path, self._lock)

    def start(self, trainer, **kwargs):
        """
        Start the process.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        self._process.start()
        self._is_open = True

    def on_iteration_end(self, trainer, **kwargs):
        """
        Update the training history.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        with self._lock:
            pickle.dump(trainer.training_history, open(self._temp_path, "wb"))

    def __del__(self):
        """
        Delete the temporary file.

        :return: None
        """
        self.close(None)

    def close(self, trainer, **kwargs):
        """
        Close the process adn delete the temporary file.

        :param trainer: The trainer.
        :type trainer: Trainer

        :return: None
        """
        if self._is_open:
            self._process.join()
            self._process.kill()
            self._process.close()
            self._is_open = False
            with self._lock:
                if os.path.exists(self._temp_path):
                    os.remove(self._temp_path)



