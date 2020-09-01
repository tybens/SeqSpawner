import numpy
import unittest2

from seqspawner import motifs
from seqspawner.sequence import simulate_background_then_motif
from seqspawner import model


class TestModel(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.pwm_motif2 = numpy.array([[1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]])

        self.alphabet = ['A', 'C', 'G', 'T']
        self.pwm = motifs.PWM(self.pwm_motif, self.alphabet)
        self.pwm2 = motifs.PWM(self.pwm_motif2, self.alphabet)
        self.length = 100
        self.loc_fn = lambda: 30
        self.background = [0.25, 0.25, 0.25, 0.25]
        self.model_test = model.Model("test1",
                                      self.pwm.alphabet,
                                      self.pwm,
                                      self.background,
                                      self.loc_fn,
                                      length=self.length)
        self.expected = "".join(['T' for i in range(self.pwm_motif.shape[1])])
        self.expected2 = "".join(['A' for i in range(self.pwm_motif.shape[1])])

    def test_sample_specific_position(self):
        # -------- the motif (7 T's) is at loc_fn as specified -------------
        observed = self.model_test.sample(1)[0]
        self.assertEqual(self.expected, observed[self.loc_fn() - 1:(self.loc_fn() + self.pwm_motif.shape[1] - 1)])
        # ---------- the length is correct -------------
        self.assertEqual(len(observed), self.length)

    def test_sample_motif_at_end_of_seq(self):
        loc_fn = lambda: 20
        length = self.pwm.size() + loc_fn() - 1
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 self.pwm,
                                 self.background,
                                 loc_fn,
                                 length=length)

        observed = model_test.sample(1)[0]
        self.assertEqual(self.expected, observed[loc_fn() - 1:])

    def test_sample_motif_at_beginning(self):
        loc_fn = lambda: 1
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 self.pwm,
                                 self.background,
                                 loc_fn,
                                 length=self.length)
        observed = model_test.sample(1)[0]

        self.assertEqual(observed[:self.pwm.size()], self.expected)

    def test_sample_all_Ts(self):
        background = [0, 0, 0, 1]
        loc_fn = lambda: numpy.random.choice(self.length - self.pwm.size()) + 1
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 self.pwm,
                                 background,
                                 loc_fn,
                                 length=self.length)
        observed = model_test.sample(1)[0]
        expected = "".join(['T' for _ in range(len(observed))])
        self.assertEqual(observed, expected)

    def test_sample_generated_outside_ValueError(self):
        loc_fn = lambda: self.length
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 self.pwm,
                                 self.background,
                                 loc_fn,
                                 length=self.length)
        with self.assertRaises(ValueError):
            model_test.sample(1)

    def test_sample_multiple_motifs_rejection(self):
        motifs_list = [self.pwm, self.pwm2]
        loc_fn = [lambda: 50, lambda: 20]
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 motifs_list,
                                 self.background,
                                 loc_fn,
                                 length=self.length)

        observed = model_test.sample(1)[0]
        expecteds = [self.expected, self.expected2]
        for i, x in enumerate(motifs_list):
            loc = loc_fn[i]() - 1
            self.assertEqual(observed[loc:loc + x.size()], expecteds[i])

    def test_sample_multiple_motifs_rejection_ValueError(self):
        motifs_list = [self.pwm, self.pwm2]
        loc_fn = [lambda: 20, lambda: 23]
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 motifs_list,
                                 self.background,
                                 loc_fn,
                                 length=self.length)

        with self.assertRaises(ValueError):
            model_test.sample(1)

    def test_sample_multiple_motifs_merge(self):
        motifs_list = [self.pwm, self.pwm2]
        loc_fn = [lambda: 20, lambda: 23]
        numpy.random.seed(30)
        model_test = model.Model("test1",
                                 self.pwm.alphabet,
                                 motifs_list,
                                 self.background,
                                 loc_fn,
                                 length=self.length,
                                 resolve_overlap='merge')
        observed = model_test.sample(1)[0]
        self.assertEqual(observed[loc_fn[0]() - 1:loc_fn[1]() - 1 + motifs_list[1].size()], 'TTTTATAAAA')

    def test_sample_variable_spacing_motif_list(self):
        numpy.random.seed(seed=30)  # only 1 nt in between for variable spacing
        background_pwm = numpy.array([[0.25], [0.25], [0.25], [0.25]])
        sep_fn = lambda: int(numpy.random.choice(4)) + 1
        repeated_bg = motifs.VariableRepeatedMotif(motifs.PWM(background_pwm, self.pwm2.alphabet), sep_fn)
        list_test = motifs.create_motif_list([self.pwm, repeated_bg, self.pwm2])
        model_test = model.Model("test2",
                                 list_test.alphabet,
                                 list_test,
                                 self.background,
                                 self.loc_fn,
                                 length=self.length)
        spacing = 15  # for seed=30
        observed = model_test.sample(1)[0]

        self.assertEqual(observed[self.loc_fn() - 1:self.loc_fn() - 1 + spacing], 'TTTTTTTCAAAAAAA')

    def test_sample_merge_encompassed(self):
        """ The second motif is small enough in comparison to the first so that the first both starts and finishes the
        merged motif.

        """
        numpy.random.seed(30)
        ctcf_pwm = motifs.PWM.from_meme_file("files/CTCF.meme")
        motifs_list = [ctcf_pwm, self.pwm]
        loc_fns = [lambda: 50, lambda: 56]

        model_test = model.Model("test2",
                                 ctcf_pwm.alphabet,
                                 motifs_list,
                                 self.background,
                                 loc_fns,
                                 length=70,
                                 resolve_overlap='merge')
        observed = model_test.sample(1)[0]

        self.assertEqual(observed, 'GCGATCTAGCAGGTTAATAAGGGATCGCACTCGCCGCGCCAGACCGTTAAGACCAGTAGGTTGCGGCCGG')


class TestMotifsClasses(unittest2.TestCase):
    """ A lot of the functionality of `motifs.py` is tested in the `TestModel` class
    """

    def test_pwm_from_meme(self):
        ctcf_pwm = motifs.PWM.from_meme_file("files/CTCF.meme")
        numpy.random.seed(seed=30)

        self.assertEqual(ctcf_pwm.sample(), 'TGGCCATCAGAGGGTACTA')


class TestSimulateBackgroundThenMotif(unittest2.TestCase):

    def setUp(self):
        self.pwm_motif = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1, 1, 1]])  # 7 nt's are T motif
        self.pos_motif = 50
        self.background_weights = [0.25, 0.25, 0.25, 0.25]
        self.expected = "".join(['T' for i in range(self.pwm_motif.shape[1])])
        self.alphabet = ['A', 'C', 'G', 'T']

    def test_simulate_background_then_motif_sevenTs(self):
        # -------- the last nt's in the seqspawner are the motif (7 T's) -------------
        observed = simulate_background_then_motif(self.pos_motif, self.pwm_motif, self.alphabet,
                                                  self.background_weights)

        self.assertEqual(self.expected, observed[self.pos_motif - 1:])
        # --------------- length -----------------
        self.assertEqual(len(observed), (self.pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_nobackground(self):
        pos_motif = 1
        observed = simulate_background_then_motif(pos_motif, self.pwm_motif, self.alphabet, self.background_weights)
        self.assertEqual(self.expected, observed[pos_motif - 1:])
        # ----------- length -----------
        self.assertEqual(len(observed), (pos_motif + self.pwm_motif.shape[1] - 1))

    def test_simulate_background_then_motif_ValueError(self):
        pos_motif = 0
        with self.assertRaises(ValueError):
            simulate_background_then_motif(pos_motif, self.pwm_motif, self.alphabet, self.background_weights)


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
        observed = model.Model._merge_motifs(self.pwm_motif[0], self.pwm_motif[1], index_of_overlap)
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
        observed = model.Model._merge_motifs(pwm_motif1, pwm_motif2, index_of_overlap)
        self.assertSequenceEqual(observed.tolist(), expected.tolist())


if __name__ == '__main__':
    unittest2.main()
