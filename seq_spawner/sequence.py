import numpy


def simulate_background_then_motif(pos_motif, motif_pwm, alphabet, background_weights):
    """Simulates DNA of background up until the motif is generated (bg then motif).

    Generates a DNA sequence of a length of position of the motif + length of the motif - 1
    That is, there is no background sequence generated after the motif is generated. This
    is to be used in `model._sample`

    Parameters
    ----------
    pos_motif : int
        The index of the location of the motif (not starting at 0)
    motif_pwm : numpy.array([[]])
        The motif as a position weight matrix.
    alphabet : list(str)
        A list of the alphabet that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities
    Returns
    -------
    list(str)
        A list of single character strings representing DNA nucleotides of the background then the motif.

    """

    seq = ""

    # background before motif
    seq += "".join(numpy.random.choice(alphabet, pos_motif - 1, p=background_weights))
    # motif
    for i in range(motif_pwm.shape[1]):
        # normalize:
        p = numpy.array(motif_pwm[:, i], dtype=numpy.float64)
        p /= p.sum()
        seq += "".join(numpy.random.choice(alphabet, 1, p=p))

    return seq


def variable_separation_combine_pwm_motifs(pwm_motif1, pwm_motif2, separation, background_weights):
    """ Combines two pwm motifs into one with background weights of a specific length separating the two

    This can be used by the users before sequence is called if a specific separation between two (or more) motifs is
    desired. It is used in `model.sample` if `self.variable_spacing_motif_indices` is not None.

    Parameters
    ----------
    pwm_motif1 : numpy.array
        The position weight matrix of the first motif (ordered from left to right)
    pwm_motif2 : numpy.array
        The position weight matrix of the second motif (ordered from left to right)
    separation : int
        The position of one motif relative to another. This is form the end of the first motif to the start of the
        second motif.
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities

    Returns
    -------
    Numpy.array
        One numpy.array in position weight matrix format for the combination of two motifs with a background sequence
        of specific length separating the two
    """

    bg_between = numpy.reshape(background_weights * separation, (pwm_motif1.shape[0], separation))
    combined_variable_separation_motif = numpy.hstack((pwm_motif1, bg_between, pwm_motif2))

    return combined_variable_separation_motif


# TODO: maybe check to see if user has input a singular fraction and if so change it to background weights
def gc_frac_to_background_weights(gc_frac):
    """Converts gc_frac to a list of weights

    Parameters
    ----------
    gc_frac : float
        fraction of G's and C's in background dna sequencing

    Returns
    -------
    list(float)
        frequency of nucleotides in the background sequence. ['A', 'C', 'G', 'T']

    """

    c_frac = gc_frac / 2
    bg_weights = [(1 / 2 - c_frac), c_frac, c_frac, (1 / 2 - c_frac)]
    return bg_weights
