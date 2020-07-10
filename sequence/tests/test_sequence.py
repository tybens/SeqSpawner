import numpy
import unittest2
import scipy.stats

from simuldna.sequence import simulate_background_then_motif
from simuldna.sequence import simulate_sequence, merge_motif
from simuldna.sequence import simulate_sequence_not_independently
from simuldna.sequence import simulate_sequence_with_single_motif


class TestSimulateSequenceWithSingleMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.length = 100
        self.pos_motif = 50
        self.background_weights = [0.25, 0.25, 0.25, 0.25]
        self.expected = ['T' for i in range(self.pwm_motif.shape[1])]

    def test_simulate_sequence_with_single_motif_specific(self):
        # -------- the motif (7 T's) is at pos_motif as specified -------------
        observed = simulate_sequence_with_single_motif(self.length, self.pos_motif, self.pwm_motif,
                                                       self.background_weights)
        self.assertEqual(self.expected, observed[self.pos_motif - 1:(self.pos_motif + self.pwm_motif.shape[1] - 1)])
        # ---------- the length is correct -------------
        self.assertEqual(len(observed), self.length)

    def test_simulate_sequence_with_single_motif_end(self):
        pos_motif = 94
        observed = simulate_sequence_with_single_motif(self.length, pos_motif, self.pwm_motif, self.background_weights)
        self.assertEqual(self.expected, observed[pos_motif - 1:])

    def test_simulate_sequence_with_single_motif_all_T(self):
        background_weights = [0, 0, 0, 1]
        observed = simulate_sequence_with_single_motif(self.length, self.pos_motif, self.pwm_motif, background_weights)
        expected = ['T' for i in range(len(observed))]
        self.assertEqual(observed, expected)

    def test_simulate_sequence_with_single_motif_ValueError(self):
        pos_motif = 95
        with self.assertRaises(ValueError):
            simulate_sequence_with_single_motif(self.length, pos_motif, self.pwm_motif, self.background_weights)


class TestSimulateBackgroundThenMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.pos_motif = 50
        self.background_weights = [0.25, 0.25, 0.25, 0.25]
        self.expected = ['T' for i in range(self.pwm_motif.shape[1])]

    def test_simulate_background_then_motif_sevenTs(self):
        # -------- the last nt's in the sequence are the motif (7 T's) -------------
        observed = simulate_background_then_motif(self.pos_motif, self.pwm_motif, self.background_weights)

        self.assertEqual(self.expected, observed[self.pos_motif - 1:])
        # --------------- length -----------------
        self.assertEqual(len(observed), (self.pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_nobackground(self):
        pos_motif = 1
        observed = simulate_background_then_motif(pos_motif, self.pwm_motif, self.background_weights)
        self.assertEqual(self.expected, observed[pos_motif - 1:])
        # ----------- length -----------
        self.assertEqual(len(observed), (pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_ValueError(self):
        pos_motif = 0
        with self.assertRaises(ValueError):
            simulate_background_then_motif(pos_motif, self.pwm_motif, self.background_weights)


class TestSimulateNotIndependently(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1]]),
                          numpy.array([[1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]])]
        self.length = 100
        self.background_weights = [0.25, 0.25, 0.25, 0.25]
        self.expectedTs = ['T' for i in range(self.pwm_motif[0].shape[1])]
        self.expectedAs = ['A' for i in range(self.pwm_motif[1].shape[1])]
        self.verbose = False

        def binom():
            # example user input function that takes no arguments and outputs a distributed position
            return scipy.stats.binom.rvs(self.length, 0.3)

        def diracdelta_40():
            return 40

        self.dist_fn_binom = [binom] * 2
        self.dist_fn_diracdelta = [diracdelta_40] * 2

    def test_simulate_not_independently_binomial(self):
        # -----verifying repetition of specific location for binomial dist_fn choice for seed=35----------
        pos_motif1_actual = 32  # seed=33
        pos_motif2_actual = 58  # seed=33
        numpy.random.seed(33)

        observed = simulate_sequence_not_independently(self.length, self.pwm_motif, self.background_weights,
                                                       self.dist_fn_binom, verbose=self.verbose)
        self.assertEqual(self.expectedAs,
                         observed[pos_motif1_actual - 1:(pos_motif1_actual + self.pwm_motif[0].shape[1] - 1)])
        self.assertEqual(self.expectedTs,
                         observed[pos_motif2_actual - 1:(pos_motif2_actual + self.pwm_motif[1].shape[1] - 1)])

    def test_simulate_not_independently_diracdelta(self):
        # -----verifying repetition of specific location for dirac delta dist_fn choice for seed=35----------
        pos_motif_1 = 40
        pos_motif_2 = 86

        observed = simulate_sequence_not_independently(self.length, self.pwm_motif, self.background_weights,
                                                       self.dist_fn_diracdelta, verbose=self.verbose)
        self.assertEqual(self.expectedAs,
                         observed[pos_motif_1 - 1:(pos_motif_1 + self.pwm_motif[0].shape[1] - 1)])
        self.assertEqual(self.expectedTs, observed[pos_motif_2 - 1:(pos_motif_2 + self.pwm_motif[1].shape[1] - 1)])

        # ----------- length ----------
        self.assertEqual(self.length, len(observed))

    def test_simulate_not_independently_ValueError(self):
        pos_motif_1 = 40
        pos_motif_2 = 40

        def diracdelta_0():
            return 0

        dist_fn = self.dist_fn_diracdelta
        dist_fn[1] = diracdelta_0

        with self.assertRaises(ValueError):
            simulate_sequence_not_independently(self.length, self.pwm_motif, self.background_weights, dist_fn,
                                                verbose=self.verbose)


class TestSimulateSequence(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1]]),
                          numpy.array([[1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]]),
                          numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]])]
        self.length = 100
        self.background_weights = [0.25, 0.25, 0.25, 0.25]
        self.verbose = False
        self.resolve_overlap = 'merge'

        def binom():
            # example user input function that takes no arguments and outputs a distributed position
            return scipy.stats.binom.rvs(self.length, 0.3)

        self.dist_fn_binom = [binom, binom, binom]

        self.total_tries = 10
        self.expectedTs = ['T' for i in range(self.pwm_motif[0].shape[1])]
        self.expectedAs = ['A' for i in range(self.pwm_motif[0].shape[1])]
        self.expectedCs = ['C' for i in range(self.pwm_motif[0].shape[1])]

    def test_simulate_sequence_three_overlap(self):
        # ---------- three overlapping motifs merged together ------------------
        numpy.random.seed(20)
        pos_motif1 = 31
        # pos_motif2 = 36 # for keepsake
        pos_motif3 = 36
        observed = simulate_sequence(self.length, self.pwm_motif, self.background_weights, self.resolve_overlap,
                                     self.dist_fn_binom,
                                     verbose=self.verbose)

        self.assertEqual('TTTTTCCCCAAA',
                         ''.join(observed[pos_motif1 - 1:pos_motif3 + self.pwm_motif[2].shape[1] - 1]))

        # ------------ length -----------------
        self.assertEqual(self.length, len(observed))

    def test_simulate_sequence_one_motif(self):
        numpy.random.seed(60)

        background_weights = [0, 0, 0, 1]
        observed = simulate_sequence(self.length, self.pwm_motif[:1], background_weights, self.resolve_overlap,
                                     self.dist_fn_binom,
                                     verbose=self.verbose)
        self.assertEqual(['T' for i in range(len(observed))], observed)

    # -------------------- REJECTION ------------------------------------

    def test_simulate_sequence_rejected_all(self):
        # ------------- 10 tries isn't enough for randomness to generate non-overlapped ( observed = [])--------- #
        numpy.random.seed(10)
        resolve_overlap = 'reject'

        observed = simulate_sequence(self.length, self.pwm_motif, self.background_weights, resolve_overlap,
                                     self.dist_fn_binom,
                                     total_tries=self.total_tries, verbose=self.verbose)
        self.assertEqual(observed, [])

    def test_simulate_sequence_solution_found(self):
        # ------------ 3 tries is enough for randomness to generate non-overlapped -------------- #
        numpy.random.seed(22)
        pos_motif1 = 19
        pos_motif2 = 28
        pos_motif3 = 38
        resolve_overlap = 'reject'
        observed = simulate_sequence(self.length, self.pwm_motif, self.background_weights, resolve_overlap,
                                     self.dist_fn_binom,
                                     total_tries=self.total_tries, verbose=self.verbose)
        self.assertEqual(self.expectedTs, observed[pos_motif1 - 1:(
                pos_motif1 + self.pwm_motif[0].shape[1] - 1)])  # shift to left by one because pos_motif* is not index
        self.assertEqual(self.expectedAs,
                         observed[pos_motif2 - 1:(pos_motif2 + self.pwm_motif[1].shape[1] - 1)])
        self.assertEqual(self.expectedCs,
                         observed[pos_motif3 - 1:(pos_motif3 + self.pwm_motif[2].shape[1] - 1)])

        """
        # ---------- Sequence could not be generated without overlap in 10 tries ------------------ #
        numpy.random.seed(seed=10)  # 10 tries are not enough
        capturedOutput = io.StringIO()  # Create StringIO object
        sys.stdout = capturedOutput  # and redirect stdout.
        # call function:
        observed = simulate_sequence(length, pwm_motif, background_weights, dist_fn, total_tries=total_tries, mu=mu,
                                          loc=loc, p=p, pos_motifs=[pos_motif], verbose=verbose)
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertEqual('Sequence could not be generated without overlap in 10 tries\n',
                         capturedOutput.getvalue())  # Now works as before.
        """

    def test_simulate_sequence_motif_end(self):
        # ---------- corner case of the motif being placed right at the end of the sequence ------- #
        pm3 = 94
        resolve_overlap = 'reject'
        numpy.random.seed(20)

        def diracdelta_94():
            return 94

        self.dist_fn_binom[2] = diracdelta_94

        observed = simulate_sequence(self.length, self.pwm_motif, self.background_weights, resolve_overlap,
                                     self.dist_fn_binom,
                                     verbose=self.verbose)

        self.assertEqual(self.expectedCs, observed[pm3 - 1:])
        # ------- length ----------
        self.assertEqual(self.length, len(observed))

    # def test_simulate_sequence_ValueError(self):  DON'T HAVE VALUEERROR FOR REJECTION SAMPLING


class TestMergeMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1]]),
                          numpy.array([[1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]])]

    def test_merge_motif_basic_overlap(self):
        index_of_overlap = 3
        observed = merge_motif(self.pwm_motif[0], self.pwm_motif[1], index_of_overlap)
        expected = numpy.array([[0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.]])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_merge_motif_complicated_overlap(self):
        pwm_motif1 = numpy.array([[0.3, 0.125, 0.25, 0.5, 0.99, 0.999],
                                  [0.15, 0.125, 0.4, 0.1, 0., 0.],
                                  [0.15, 0.125, 0.25, 0.125, 0., 0.],
                                  [0.4, 0.125, 0.1, 0.275, 0.01, 0.001]])
        pwm_motif2 = numpy.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.],
                                  [0.2, 0.3, 0.2, 0.2, 0.1, 0.1, 0.],
                                  [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.],
                                  [0.1, 0., 0.1, 0.1, 0.2, 0.2, 1.]])
        expected = numpy.array([[0.3, 0.1125, 0.225, 0.4, 0.695, 0.7495, 0.6, 0.],
                                [0.15, 0.1625, 0.35, 0.15, 0.1, 0.05, 0.1, 0.],
                                [0.15, 0.3625, 0.375, 0.2625, 0.15, 0.1, 0.1, 0.],
                                [0.4, 0.1125, 0.05, 0.1875, 0.055, 0.1005, 0.2, 1.]])
        index_of_overlap = 2
        observed = merge_motif(pwm_motif1, pwm_motif2, index_of_overlap)
        self.assertSequenceEqual(observed.tolist(), expected.tolist())


if __name__ == '__main__':
    unittest2.main()
