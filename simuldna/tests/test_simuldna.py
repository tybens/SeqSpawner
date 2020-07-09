import numpy as np
import unittest2

from simuldna.sequence import simulate_background_then_motif
from simuldna.sequence import simuldna_indep_merge, merge_motif
from simuldna.sequence import simuldna_indep_rejection
from simuldna.sequence import simulate_sequence_not_independently
from simuldna.sequence import simulate_sequence_with_single_motif


class TestSimulateSequenceWithSingleMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.length = 100
        self.pos_motif = 50
        self.gc_frac = 0.5

    def test_simulate_sequence_with_single_motif_specific(self):
        # -------- the motif (7 T's) is at pos_motif as specified -------------
        sample = simulate_sequence_with_single_motif(self.length, self.pos_motif, self.pwm_motif, self.gc_frac)
        self.assertIn('TTTTTTT', ''.join(sample[self.pos_motif - 1:(self.pos_motif + self.pwm_motif.shape[1] - 1)]))
        # ---------- the length is correct -------------
        self.assertEqual(len(sample), self.length)

    def test_simulate_sequence_with_single_motif_end(self):
        pos_motif = 94
        sample = simulate_sequence_with_single_motif(self.length, pos_motif, self.pwm_motif, self.gc_frac)
        self.assertIn('TTTTTTT', ''.join(sample[pos_motif - 1:]))

    def test_simulate_sequence_with_single_motif_ValueError(self):
        pos_motif = 95
        with self.assertRaises(ValueError):
            simulate_sequence_with_single_motif(self.length, pos_motif, self.pwm_motif, self.gc_frac)


class TestSimulateBackgroundThenMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.pos_motif = 50
        self.gc_frac = 0.5

    def test_simulate_background_then_motif_sevenTs(self):
        # -------- the last nt's in the sequence are the motif (7 T's) -------------
        sample = simulate_background_then_motif(self.pos_motif, self.pwm_motif, self.gc_frac)
        self.assertIn('TTTTTTT', ''.join(sample[self.pos_motif - 1:]))
        # --------------- length -----------------
        self.assertEqual(len(sample), (self.pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_nobackground(self):
        pos_motif = 1
        sample = simulate_background_then_motif(pos_motif, self.pwm_motif, self.gc_frac)
        self.assertIn('TTTTTTT', ''.join(sample[pos_motif - 1:]))
        # ----------- length -----------
        self.assertEqual(len(sample), (pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_ValueError(self):
        pos_motif = 0
        with self.assertRaises(ValueError):
            simulate_background_then_motif(pos_motif, self.pwm_motif, self.gc_frac)


class TestSimuldnaNotIndepDist(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [np.array([[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1]]),
                          np.array([[1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])]
        self.length = 100
        self.gc_frac = 0.5
        self.mu = 0.5
        self.p = 0.7
        self.loc = 0

    def test_simuldna_notindep_dist_binomial(self):
        pos_motif_1 = 40  # input into the poss_motif list for binomial dists
        pos_motif_2 = 87
        distribution = 'binomial'
        # -----verifying repetition of specific location for binomial distribution choice for seed=35----------
        pos_motif1_actual = 44  # seed=35
        pos_motif2_actual = 87  # seed=35
        np.random.seed(35)

        sample = simulate_sequence_not_independently(self.length, self.pwm_motif, self.gc_frac, distribution, mu=self.mu,
                                                     loc=self.loc, p=self.p,
                                                     pos_motifs=[pos_motif_1, pos_motif_2], verbose=True)
        self.assertIn('TTTTTTT',
                      ''.join(sample[pos_motif1_actual - 1:(pos_motif1_actual + self.pwm_motif[0].shape[1] - 1)]))
        self.assertIn('AAAAAAA',
                      ''.join(sample[pos_motif2_actual - 1:(pos_motif2_actual + self.pwm_motif[1].shape[1] - 1)]))

    def test_simuldna_notindep_dist_diracdelta(self):
        # -----verifying repetition of specific location for dirac delta distribution choice for seed=35----------
        pos_motif_1 = 40
        pos_motif_2 = 80
        distribution = 'diracdelta'
        sample = simulate_sequence_not_independently(self.length, self.pwm_motif, self.gc_frac, distribution, mu=self.mu,
                                                     loc=self.loc, p=self.p,
                                                     pos_motifs=[pos_motif_1, pos_motif_2], verbose=True)
        self.assertIn('TTTTTTT', ''.join(sample[pos_motif_1 - 1:(pos_motif_1 + self.pwm_motif[0].shape[1] - 1)]))
        self.assertIn('AAAAAAA', ''.join(sample[pos_motif_2 - 1:(pos_motif_2 + self.pwm_motif[1].shape[1] - 1)]))

        # ----------- length ----------
        self.assertEqual(self.length, len(sample))

    def test_simuldna_notindep_dist_ValueError(self):
        pos_motif_1 = 40
        pos_motif_2 = 40
        distribution = 'diracdelta'
        with self.assertRaises(Exception):
            simulate_sequence_not_independently(self.length, self.pwm_motif, self.gc_frac, distribution, mu=self.mu,
                                                loc=self.loc, p=self.p,
                                                pos_motifs=[pos_motif_1, pos_motif_2], verbose=True)


class TestSimuldnaIndepRejection(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [np.array([[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1]]),
                          np.array([[1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]),
                          np.array([[0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])]
        self.length = 100
        self.gc_frac = 0.5
        self.distributions = ['logarithmic', 'binomial', 'planck']
        self.pos_motif = 20
        self.total_tries = 10
        self.lam = 0.5
        self.mu = 0.5
        self.p = 0.7
        self.loc = 0
        self.verbose = False

    def test_simuldna_indep_rejection_rejected_all(self):
        # ------------- 10 tries isn't enough for randomness to generate non-overlapped ( sample = [])--------- #
        np.random.seed(10)
        sample = simuldna_indep_rejection(self.length, self.pwm_motif, self.gc_frac, self.distributions,
                                          total_tries=self.total_tries, mu=self.mu, lam=self.lam,
                                          loc=self.loc, p=self.p, pos_motifs=[self.pos_motif], verbose=self.verbose)
        self.assertEqual(sample, [])

    def test_simuldna_indep_rejection_solution_found(self):
        # ------------ 4 tries is enough for randomness to generate non-overlapped -------------- #
        np.random.seed(12)
        pos_motif1 = 2
        pos_motif2 = 19
        pos_motif3 = 10
        sample = simuldna_indep_rejection(self.length, self.pwm_motif, self.gc_frac, self.distributions,
                                          total_tries=self.total_tries, mu=self.mu, lam=self.lam,
                                          loc=self.loc, p=self.p, pos_motifs=[self.pos_motif], verbose=self.verbose)
        self.assertIn('TTTTTTT', ''.join(sample[pos_motif1 - 1:(
                pos_motif1 + self.pwm_motif[0].shape[1] - 1)]))  # shift to left by one because pos_motif* is not index
        self.assertIn('AAAAAAA', ''.join(sample[pos_motif2 - 1:(pos_motif2 + self.pwm_motif[1].shape[1] - 1)]))
        self.assertIn('CCCCCCC', ''.join(sample[pos_motif3 - 1:(pos_motif3 + self.pwm_motif[2].shape[1] - 1)]))

        """
        # ---------- Sequence could not be generated without overlap in 10 tries ------------------ #
        np.random.seed(seed=10)  # 10 tries are not enough
        capturedOutput = io.StringIO()  # Create StringIO object
        sys.stdout = capturedOutput  # and redirect stdout.
        # call function:
        sample = simuldna_indep_rejection(length, pwm_motif, background_weights, distributions, total_tries=total_tries, mu=mu,
                                          loc=loc, p=p, pos_motifs=[pos_motif], verbose=verbose)
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertEqual('Sequence could not be generated without overlap in 10 tries\n',
                         capturedOutput.getvalue())  # Now works as before.
        """

    def test_simuldna_indep_rejection_motif_end(self):
        # ---------- corner case of the motif being placed right at the end of the sequence ------- #
        pm1 = 10
        pm2 = 25
        pm3 = 94
        distributions = ['diracdelta', 'diracdelta', 'diracdelta']
        sample = simuldna_indep_rejection(self.length, self.pwm_motif, self.gc_frac, distributions,
                                          total_tries=self.total_tries, mu=self.mu, lam=self.lam,
                                          loc=self.loc, p=self.p, pos_motifs=[pm1, pm2, pm3], verbose=self.verbose)
        self.assertIn('CCCCCCC', ''.join(sample[pm3 - 1:]))

        # ------- length ----------
        self.assertEqual(self.length, len(sample))
    # def test_simuldna_indep_rejection_ValueError(self):  DON'T HAVE VALUEERROR FOR REJECTION SAMPLING


class TestSimuldnaIndepMerge(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = [np.array([[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1]]),
                          np.array([[1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]),
                          np.array([[0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])]
        self.length = 100
        self.gc_frac = 0.5
        self.distributions = ['binomial', 'binomial', 'binomial']
        self.total_tries = 10
        self.mu = 0.5
        self.p = 0.7
        self.loc = 0
        self.verbose = False

    def test_merge_motif_basic_overlap(self):
        index_of_overlap = 3
        observed = merge_motif(self.pwm_motif[0], self.pwm_motif[1], index_of_overlap)
        expected = np.array([[0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.]])
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_merge_motif_complicated_overlap(self):
        pwm_motif1 = np.array([[0.3, 0.125, 0.25, 0.5, 0.99, 0.999],
                               [0.15, 0.125, 0.4, 0.1, 0., 0.],
                               [0.15, 0.125, 0.25, 0.125, 0., 0.],
                               [0.4, 0.125, 0.1, 0.275, 0.01, 0.001]])
        pwm_motif2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.],
                               [0.2, 0.3, 0.2, 0.2, 0.1, 0.1, 0.],
                               [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.],
                               [0.1, 0., 0.1, 0.1, 0.2, 0.2, 1.]])
        expected = np.array([[0.3, 0.1125, 0.225, 0.4, 0.695, 0.7495, 0.6, 0.],
                             [0.15, 0.1625, 0.35, 0.15, 0.1, 0.05, 0.1, 0.],
                             [0.15, 0.3625, 0.375, 0.2625, 0.15, 0.1, 0.1, 0.],
                             [0.4, 0.1125, 0.05, 0.1875, 0.055, 0.1005, 0.2, 1.]])
        index_of_overlap = 2
        observed = merge_motif(pwm_motif1, pwm_motif2, index_of_overlap)
        self.assertSequenceEqual(observed.tolist(), expected.tolist())

    def test_simuldna_indep_merge_three_overlap(self):
        pm1 = 20
        pm2 = 22
        pm3 = 25

        # ---------- three overlapping motifs merged together ------------------
        np.random.seed(seed=20)
        pos_motif1 = 21
        # pos_motif2 = 27  # for keepsake
        pos_motif3 = 30
        sample = simuldna_indep_merge(self.length, self.pwm_motif, self.gc_frac, self.distributions, mu=self.mu,
                                      loc=self.loc, p=self.p,
                                      pos_motifs=[pm1, pm2, pm3], verbose=self.verbose)
        self.assertIn('TTTTTTAAAAACACCC', ''.join(sample[pos_motif1 - 1:pos_motif3 + self.pwm_motif[2].shape[1] - 1]))

        # ------------ length -----------------
        self.assertEqual(self.length, len(sample))

    def test_simuldna_indep_merge_three_nonoverlap(self):
        pm1 = 20
        pm2 = 29
        pm3 = 60

        np.random.seed(20)
        pos_motif1 = 21
        sample = simuldna_indep_merge(self.length, self.pwm_motif, self.gc_frac, self.distributions, mu=self.mu,
                                      loc=self.loc, p=self.p,
                                      pos_motifs=[pm1, pm2, pm3], verbose=self.verbose)
        self.assertIn('TTTTTTT', ''.join(sample[pos_motif1 - 1:pos_motif1 + self.pwm_motif[0].shape[1] - 1]))


if __name__ == '__main__':
    unittest2.main()
