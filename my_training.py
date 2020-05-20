from keras.models import Model
from keras import callbacks as cbks
import warnings
from keras.utils import Sequence
from keras.utils import GeneratorEnqueuer
from keras.utils import OrderedEnqueuer
from my_callback import My_Callback

class My_Model(Model):
    def _get_deduped_metrics_names(self):
        out_labels = self.metrics_names
        # Rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows).
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        return deduped_out_labels

    def fit_generator(self, generator,
                      steps_per_epoch,
                      epochs=1,
                      verbose=1,
                      args=None,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      initial_epoch=0):
        wait_time = 0.01  # in seconds
        epoch = initial_epoch

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__') or
                   isinstance(validation_data, Sequence))
        if val_gen and not validation_steps:
            raise ValueError('When using a generator for validation data, '
                             'you must specify a value for '
                             '`validation_steps`.')

        # Prepare display labels.
        out_labels = self._get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger(count_mode='steps')]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            print("in do_validation")
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('`validation_data` should be a tuple '
                                 '`(val_x, val_y, val_sample_weight)` '
                                 'or `(val_x, val_y)`. Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y, val_sample_weight)
            val_data = val_x + val_y + val_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                val_data += [0.]
            for cbk in callbacks:
                cbk.validation_data = val_data
        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator,
                                           use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            callback_model.stop_training = False
            while epoch < epochs:
                callbacks.on_epoch_begin(epoch)
                # My_Callback.on_epoch_end(callbacks, epoch, validation_data)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    # build batch logs
                    batch_logs = {}
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    callbacks.on_batch_begin(batch_index, batch_logs)

                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)
                    # My_Callback.on_epoch_end(callbacks, epoch, validation_data)

                    # Construct epoch logs.
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch and do_validation:
                        if val_gen:
                            val_outs = self.evaluate_generator(
                                validation_data,
                                validation_steps,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                use_multiprocessing=use_multiprocessing)
                        else:
                            # No need for try/except because
                            # data has already been validated.
                            val_outs = self.evaluate(
                                val_x, val_y,
                                batch_size=batch_size,
                                sample_weight=val_sample_weights,
                                verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(epoch)
                My_Callback.my_on_epoch_end(callbacks, epoch, validation_data, args)
                epoch += 1
                if callback_model.stop_training:
                    break

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        callbacks.on_train_end()
        return self.history