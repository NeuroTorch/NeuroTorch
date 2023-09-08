from copy import deepcopy
from typing import Dict, List, NamedTuple, Optional

# from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .buffers import ReplayBuffer


class CompletionCriteria(NamedTuple):
    """
    Completion criteria for a lesson.
    """
    measure: str
    min_lesson_length: int
    threshold: float

    @staticmethod
    def default_criteria() -> 'CompletionCriteria':
        return CompletionCriteria(measure='Rewards', min_lesson_length=1, threshold=0.9)


class Lesson:
    UNPICKLABLE_ATTRIBUTES = ['_teacher', '_channel']

    def __init__(
            self,
            name,
            channel,  #: EnvironmentParametersChannel,
            params: Dict[str, float],
            completion_criteria: CompletionCriteria = CompletionCriteria.default_criteria(),
            teacher=None,
            teacher_strength : Optional[float] = None
    ):
        # assert teacher is None or hasattr(teacher, 'buffer'), 'Teacher must have a replay buffer.'
        self.name = name
        self.completion_criteria = completion_criteria
        self._channel = channel
        self.params = params
        self._num_iterations = 0
        self._teacher = teacher
        self.teacher_strength = teacher_strength
        self._result = None

    @property
    def teacher(self):
        return self._teacher

    @teacher.setter
    def teacher(self, teacher):
        self._teacher = teacher

    @property
    def teacher_buffer(self) -> Optional[ReplayBuffer]:
        """
        Returns the replay buffer for the lesson.
        """
        if self._teacher is not None:
            buffer = self._teacher.buffer
            assert isinstance(buffer, ReplayBuffer), 'Teacher must have a replay buffer.'
            return self._teacher.buffer
        return None

    @property
    def is_completed(self):
        """
        Returns True if the lesson is completed, False otherwise.
        """
        return self._result is not None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    def __getstate__(self):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.UNPICKLABLE_ATTRIBUTES
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def set_result(self, result):
        self._result = result
        if self._teacher is not None:
            self._teacher.discard_buffer()

    def start(self):
        """
        Starts the lesson.
        """
        for key, value in self.params.items():
            self._channel.set_float_parameter(key, value)

    def on_iteration_end(self, metrics: Dict[str, float]) -> bool:
        """
        Called when an iteration ends.
        """
        self._num_iterations += 1
        return self.check_completion_criteria(metrics)

    def check_completion_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        Checks if the lesson is completed.
        """
        if self.completion_criteria.min_lesson_length > self._num_iterations:
            return False
        result = metrics[self.completion_criteria.measure]
        if result >= self.completion_criteria.threshold:
            self.set_result(result)
        return self.is_completed


class CurriculumEndIterationOutput(NamedTuple):
    """
    Output of the curriculum when the end of an iteration is reached.
    """
    messages: Dict[str, str]
    lesson_completed: bool


class Curriculum:
    def __init__(self, name: str = "Curriculum", description: str = "", lessons: List[Lesson] = None):
        self.name = name
        self.description = description
        self._lessons = [] if lessons is None else lessons
        self._current_lesson_idx = 0

    @property
    def is_completed(self) -> bool:
        """
        Returns True if the curriculum is completed, False otherwise.
        """
        return self._current_lesson_idx >= len(self._lessons)

    @property
    def current_lesson(self) -> Optional[Lesson]:
        """
        Returns the current lesson.
        """
        if self.is_completed:
            return None
        return self._lessons[self._current_lesson_idx]

    @property
    def map_repr(self) -> Dict[str, str]:
        progress = f"{100 * self._current_lesson_idx / (len(self._lessons)-1):.1f} %"
        if self.is_completed:
            return {self.name: f'(lesson: completed) 100%'}
        return {self.name: f'(lesson: {self.current_lesson.name}) {progress}'}

    @property
    def teachers(self):
        return [lesson.teacher for lesson in self._lessons]

    @property
    def channels(self):
        return [lesson.channel for lesson in self._lessons]

    @property
    def lessons(self):
        return self._lessons

    def __getitem__(self, item) -> Lesson:
        if isinstance(item, int):
            return self._lessons[item]
        elif isinstance(item, str):
            return next(filter(lambda l: l.name == item, self._lessons))
        else:
            raise TypeError(f"Invalid type for curriculum indexing: {type(item)}")

    def add_lesson(self, lesson: Lesson):
        self._lessons.append(lesson)

    def __str__(self):
        map_repr = self.map_repr
        return f'{self.name} {map_repr[self.name]}'

    def __repr__(self):
        return str(self)

    @property
    def teacher_buffer(self) -> Optional[ReplayBuffer]:
        """
        Returns the current teacher buffer.
        """
        if self.is_completed:
            return None
        return self._lessons[self._current_lesson_idx].teacher_buffer

    def on_iteration_start(self):
        """
        Called when an iteration starts.
        """
        if not self.is_completed:
            self.current_lesson.start()

    def on_iteration_end(self, metrics: Dict[str, float]) -> CurriculumEndIterationOutput:
        """
        Called when an iteration ends.
        """
        lesson_is_completed = False
        if not self.is_completed:
            self.current_lesson.on_iteration_end(metrics)
            if self.current_lesson.is_completed:
                self._current_lesson_idx += 1
                lesson_is_completed = True
        return CurriculumEndIterationOutput(messages=self.map_repr, lesson_completed=lesson_is_completed)

    def update_teachers(self, teachers: List):
        assert len(teachers) == len(self._lessons), 'Number of teachers must match number of lessons.'
        for lesson, teacher in zip(self._lessons, teachers):
            lesson.teacher = teacher

    def update_channels(self, channels: List):
        assert len(channels) == len(self._lessons), 'Number of channels must match number of lessons.'
        for lesson, channel in zip(self._lessons, channels):
            lesson._channel = channel

    def update_teachers_and_channels(self, other: 'Curriculum'):
        self.update_teachers(other.teachers)
        self.update_channels(other.channels)


