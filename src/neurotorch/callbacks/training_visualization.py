import os.path
import pickle
import time
import multiprocessing as mp
from typing import Union

from matplotlib import pyplot as plt

from .base_callback import BaseCallback


class _VisualizerProcess(mp.Process):
	"""
	Process to load a training history from pickle file and plot it. The process will update the plot every `update_dt`
	seconds and will close when join method is called.
	"""
	def __init__(self, history_path: str, lock: Union[mp.Lock, mp.RLock], update_dt: float = 1.0):
		"""
		Create a new process to plot the training history.

		:param history_path: The path to the pickle file containing the training history.
		:param lock: The lock to use to protect the training history.
		:param update_dt: The time between two updates of the plot.
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

		self.fig, self.axes, self.lines = self._training_history.create_plot(show=False, close=False)
		plt.pause(self._update_dt)
		while not self._close_event.is_set():
			self.update_history()
			self.fig, self.axes, self.lines = self._training_history.update_fig(self.fig, self.axes, self.lines)
			plt.pause(self._update_dt)

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
		Set le close event.
		Join le processus courant.
		"""
		self._close_event.set()
		return super().join(*args, **kwargs)


class TrainingHistoryVisualizationCallback(BaseCallback):
	"""
	This callback is used to visualize the training history in real time.
	"""

	def __init__(
			self,
			temp_folder: str = '~/temp/'
	):
		"""
		Create a new callback to visualize the training history.
		:param temp_folder: The folder where to save the training history.
		"""
		super().__init__()
		self._is_open = False
		os.makedirs(temp_folder, exist_ok=True)
		self._temp_path = os.path.join(
			os.path.expanduser(temp_folder), f'training_visualization{time.time_ns()}.pkl'
		)
		self._lock = mp.Lock()
		self._process = _VisualizerProcess(self._temp_path, self._lock)

	def start(self, trainer):
		"""
		Start the process.
		:param trainer: The trainer.
		:return: None
		"""
		self._process.start()
		self._is_open = True

	def on_iteration_end(self, trainer):
		"""
		Update the training history.
		:param trainer: The trainer.
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

	def close(self, trainer):
		"""
		Close the process adn delete the temporary file.
		:param trainer: The trainer.
		:return: None
		"""
		if self._is_open:
			self._process.join()
			self._process.kill()
			self._process.close()
			self._is_open = False
			with self._lock:
				os.remove(self._temp_path)


