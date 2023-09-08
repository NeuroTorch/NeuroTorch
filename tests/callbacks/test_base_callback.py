import unittest

from neurotorch.callbacks.base_callback import BaseCallback, CallbacksList
from ..mocks import MockTrainer, MockCallback


class TestBaseCallback(unittest.TestCase):
    def setUp(self):
        initial_counter = MockCallback.instance_counter
        self.callbacks = CallbacksList()
        for i in range(10):
            self.callbacks.append(MockCallback(priority=i, name=f'callback{i}', save_state=True, load_state=True))

        self.assertEqual(MockCallback.instance_counter, initial_counter+len(self.callbacks))

        self.trainer = MockTrainer()
        self.trainer.callbacks = self.callbacks

    def test_priority(self):
        # Check that the callbacks are sorted by priority
        for i in range(1, len(self.callbacks)):
            self.assertGreater(self.callbacks[i].priority, self.callbacks[i - 1].priority)

    def test_remove(self):
        # Check that the remove method works
        c0, c1 = self.callbacks[0], self.callbacks[1]
        self.callbacks.remove(c0)
        self.assertEqual(len(self.callbacks), 9)
        self.assertEqual(self.callbacks[0].name, 'callback1')
        self.assertEqual(self.callbacks[0].priority, 1)

        self.callbacks.remove(c1)
        self.assertEqual(len(self.callbacks), 8)
        self.assertEqual(self.callbacks[0].name, 'callback2')
        self.assertEqual(self.callbacks[0].priority, 2)

        for i in range(1, len(self.callbacks)):
            self.assertGreater(self.callbacks[i].priority, self.callbacks[i - 1].priority)

        self.callbacks.append(c0)
        self.callbacks.append(c1)

        for i in range(1, len(self.callbacks)):
            self.assertGreater(self.callbacks[i].priority, self.callbacks[i - 1].priority)

    def test_call_counter(self):
        # Check that the call counter is correct

        nb_call_per_itr = {
            1: [
                'on_iteration_begin', 'on_iteration_end', 'on_train_begin', 'on_train_end',
                'on_validation_begin', 'on_validation_end'
            ],
            2: ['on_epoch_begin', 'on_epoch_end', 'on_batch_begin', 'on_batch_end']
        }
        for callback in self.callbacks:
            for value, mtds in nb_call_per_itr.items():
                for mtd in mtds:
                    self.assertEqual(callback.call_mthds_counter[mtd], 0)

        n_iterations = 2
        self.trainer.train(n_iterations=n_iterations)
        for callback in self.callbacks:
            for value, mtds in nb_call_per_itr.items():
                for mtd in mtds:
                    self.assertEqual(callback.call_mthds_counter[mtd], value * n_iterations)

        for callback in self.callbacks:
            for mtd in ["start", "close"]:
                self.assertEqual(callback.call_mthds_counter[mtd], 1)

	




