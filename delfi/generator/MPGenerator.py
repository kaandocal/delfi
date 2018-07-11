import abc
import multiprocessing as mp
import numpy as np

from delfi.generator.Default import Default
from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar


class Worker(mp.Process):
    def __init__(self, n, queue, conn, model, summary, seed=None, verbose=False):
        super().__init__()
        self.n = n
        self.queue = queue
        self.verbose = verbose
        self.conn = conn
        self.model = model
        self.summary = summary
        self.rng = np.random.RandomState(seed=seed)

    def update(self, i):
        self.queue.put(i)

    def run(self):
        self.log("Starting worker")
        while True:
            try:
                self.log("Listening")
                params_batch, sources_batch = self.conn.recv()
            except EOFError:
                self.log("Leaving")
                break
            
            if len(params_batch) == 0:
                self.log("Skipping")
                self.conn.send(([], []))
                continue

            # run forward model for all params, each n_reps times
            self.log("Received data of size {}".format(len(params_batch)))
            result = self.model.gen(params_batch, pbar=self)

            stats, params, sources = self.process_batch(params_batch, sources_batch, result)

            self.log("Sending data")
            self.queue.put((stats, params, sources))
            self.log("Done")

    def process_batch(self, params_batch, sources_batch, result):
        ret_stats = []
        ret_params = []
        ret_sources = []

        # for every datum in data, check validity
        params_data_valid = []  # list of params with valid data
        sources_data_valid = []
        data_valid = []  # list of lists containing n_reps dicts with data

        for param, source, datum in zip(params_batch, sources_batch, result):
            data_valid.append(datum)
            params_data_valid.append(param)
            sources_data_valid.append(source)

        # for every data in data, calculate summary stats
        for param, source, datum in zip(params_data_valid, sources_data_valid, data_valid):
            # calculate summary statistics
            sum_stats = self.summary.calc(datum)  # n_reps x dim stats

            ret_stats.append(sum_stats)
            ret_params.append(param)
            ret_sources.append(source)

        return ret_stats, ret_params, ret_sources
    
    def log(self, msg):
        if self.verbose:
            print("Worker {}: {}".format(self.n, msg))


class MPGenerator(Default):
    def __init__(self, models, prior, summary, seed=None, verbose=False):
        """Generator

        Parameters
        ----------
        model : Simulator instance
            Forward model
        prior : Distribution or Mixture instance
            Prior over parameters
        summary : SummaryStats instance
            Summary statistics

        Attributes
        ----------
        proposal : None or Distribution or Mixture instance
            Proposal prior over parameters. If specified, will generate
            samples given parameters drawn from proposal distribution rather
            than samples drawn from prior when `gen` is called.
        """
        super().__init__(model=None, prior=prior, summary=summary, seed=None)
        self.verbose = verbose
        pipes = [ mp.Pipe(duplex=True) for m in models ]
        self.queue = mp.Queue()
        self.workers = [ Worker(i, self.queue, pipes[i][1], models[i], summary, seed=self.rng.randint(low=0,high=2**31), verbose=verbose) for i in range(len(models)) ]
        self.pipes = [ p[0] for p in pipes ]

        self.log("Starting workers")
        for w in self.workers:
            w.start()

        self.log("Done")

    def iterate_minibatches(self, params, sources, minibatch=50):
        n_samples = len(params)

        for i in range(0, n_samples - minibatch+1, minibatch):
            yield params[i:i + minibatch], sources[i:i + minibatch]

        rem_i = n_samples - (n_samples % minibatch)
        if rem_i != n_samples:
            yield params[rem_i:], sources[rem_i:]

    def gen(self, n_samples, n_reps=1, skip_feedback=False, prior_mixin=0, verbose=True, **kwargs):
        """Draw parameters and run forward model

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_reps: int
            Number of repetitions per parameter sample
        skip_feedback: bool
            If True, feedback checks on params, data and sum stats are skipped
        verbose : bool or str
            If False, will not display progress bars. If a string is passed,
            it will be appended to the description of the progress bar.

        Returns
        -------
        params : n_samples x n_reps x n_params
            Parameters
        stats : n_samples x n_reps x n_summary
            Summary statistics of data
        """
        assert n_reps == 1, 'n_reps > 1 is not yet supported'

        params, sources = self.draw_params(n_samples=n_samples,
                                           skip_feedback=skip_feedback, 
                                           prior_mixin=prior_mixin,
                                           verbose = verbose)

        return self.run_model(params, sources, skip_feedback=skip_feedback, verbose=verbose, **kwargs)

    def run_model(self, params, sources, minibatch=50, skip_feedback=False, keep_data=True, verbose=False):
        # Run forward model for params (in batches)
        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=len(params))
            desc = 'Run simulations '
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        final_params = []
        final_sources = []
        final_stats = []  # list of summary stats
        minibatches = self.iterate_minibatches(params, sources, minibatch)
        done = False
        with pbar:
            while not done:
                active_list = []
                for w, p in zip(self.workers, self.pipes):
                    try:
                        params_batch, sources_batch = next(minibatches)
                    except StopIteration:
                        done = True
                        break

                    active_list.append((w,p))
                    self.log("Dispatching to worker (len = {})".format(len(params_batch)))
                    p.send((params_batch, sources_batch))
                    self.log("Done")

                n_remaining = len(active_list)
                while n_remaining > 0:
                    self.log("Listening to worker")
                    msg = self.queue.get()
                    if type(msg) == int:
                        self.log("Received int")
                        pbar.update(msg)
                    elif type(msg) == tuple:
                        self.log("Received results")
                        stats, params, sources = self.filter_data(*msg, skip_feedback=skip_feedback)
                        final_stats += stats
                        final_params += params
                        final_sources += sources
                        n_remaining -= 1
                    else:
                        self.log("Warning: Received unknown message of type {}".format(type(msg)))

        # TODO: for n_reps > 1 duplicate params; reshape stats array

        # n_samples x n_reps x dim theta
        params = np.array(final_params)
        sources = np.array(final_sources)

        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze(axis=1)

        return params, stats, sources

    def filter_data(self, stats, params, sources, skip_feedback=False):
        if skip_feedback == True:
            return stats, params, sources

        ret_stats = []
        ret_params = []
        ret_sources = []

        for stat, param, source in zip(stats, params, sources):
            response = self._feedback_summary_stats(stat)
            if response == 'accept':
                ret_stats.append(stat)
                ret_stats.append(params)
                ret_stats.append(sources)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')
        
        return ret_stats, ret_params, ret_sources

    def log(self, msg):
        if self.verbose:
            print("Parent: {}".format(msg))

    def __del__(self):
        self.log("Closing")
        for w, p in zip(self.workers, self.pipes):
            self.log("Closing pipe")
            p.close()

        for w in self.workers:
            self.log("Joining process")
            w.join(timeout=1)
            w.terminate()
