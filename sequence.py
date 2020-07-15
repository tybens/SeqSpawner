import numpy
import operator


def simulate_sequence_with_single_motif(length, pos_motif, pwm_motif, emissions, background_weights):
    """Simulates a dna sequence with one embedded motif at a specific location.

    Parameters
    ----------
    length : int
        The desired total length of the sequence
    pos_motif : int
        The index of the location of the motif (not starting at 0)
    pwm_motif : numpy.array
        The motif as a numpy.array in position weight matrix format for dna
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities

    Returns
    -------
    list(str)
        The genomic sequence of length = length as a list of single character strings with one motif embedded at a
        specific location

    """
    seq = []

    # background before motif
    seq.extend(numpy.random.choice(emissions, pos_motif - 1, p=background_weights))

    # motif
    len_motif = pwm_motif.shape[1]
    for i in range(len_motif):
        seq.extend(numpy.random.choice(emissions, p=pwm_motif[:, i]))

    # background after
    seq.extend(numpy.random.choice(nts, length - (pos_motif + len_motif - 1), p=background_weights))

    return seq


def simulate_background_then_motif(pos_motif, pwm_motif, emissions, background_weights):
    """Simulates DNA of background up until the motif is generated (bg then motif).
    
    Generates a DNA sequence of a length of position of the motif + length of the motif - 1
    That is, there is no background sequence generated after the motif is generated. This
    is to be used in simulate_sequence.
    
    Parameters
    ----------
    pos_motif : int
        The index of the location of the motif (not starting at 0)
    pwm_motif : numpy.array
        The motif as a numpy.array in position weight matrix format. For dna a column is ['A', 'C', 'G', 'T']
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities
    Returns
    -------
    list(str)
        A list of single character strings representing DNA nucleotides of the background then the motif.

    """

    seq = []

    # background before motif
    seq.extend(numpy.random.choice(emissions, pos_motif - 1, p=background_weights))

    # motif 
    len_motif = pwm_motif.shape[1]
    for i in range(len_motif):
        seq.extend(numpy.random.choice(emissions, p=pwm_motif[:, i]))

    return seq


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


def simulate_sequence_not_independently(length, pwm_motif, emissions, background_weights, dist_fn, verbose=False):
    """ Simulates DNA with multiple motifs at locations distributed relative to each previous motif location.

    The motif locations are sampled using the user-input distribution function; however, the output position from dist_fn
    will be added to the current length of the sequence. That is, the probability distribution is shifted by the
    previous motif position + the previous motif's length.

    Parameters
    ----------
    length : int
        The desired total length of the sequence
    dist_fn : list(function)
        A list of functions for each motif that when called outputs a random int position for the motif.
        For an example function: def binom():
                                    return scipy.stats.binom.rvs(...)
    verbose : bool, optional
        Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each
        iteration)
    pwm_motif : list(numpy.array)
        The motifs as a list of numpy.arrays in position weight matrix format for dna ['A', 'C', 'G', 'T']. Can be no
        motifs with an input of an empty list [].
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities

    Returns
    -------
    list(str)
        The genomic sequence of length = length as a list of single character strings.
    """

    numpy.random.shuffle(pwm_motif)  # shuffle the motifs to be rid of ordering bias
    num_motifs = len(pwm_motif)  # total number of motifs to be embedded
    sample = []  # initialize empty sequence list

    for i in range(num_motifs):

        pos_motif = dist_fn[i]()  # scipy.stats.rvs(...)

        if verbose:
            print(" position of motif {}: ".format(str(i + 1)) + str(pos_motif + len(sample)))

        if len(sample) >= (pos_motif + len(sample)) or (pos_motif + len(sample)) >= length - pwm_motif[i].shape[1]:
            raise (ValueError("motif number {} generated outside of DNA sequence".format(str(i + 1))))

        sample.extend(simulate_background_then_motif(pos_motif, pwm_motif[i], emissions, background_weights))

    # background sequence after the last motif
    len_sample = len(sample)
    sample.extend(numpy.random.choice(emissions, length - len_sample, p=background_weights))
    return sample


def merge_motif(pwm_motif1, pwm_motif2, index_of_overlap):
    """ Merges two overlapping motifs.

    Combines the positional weights of each nucleotide and outputs a combined matrix. For example:
    With pwm_motif1 = [A, C, T, G] and pwm_motif2 = [A, A, A, A] and an index_of_overlap = 3:
    As shown here,
           [A, A, A, A]
    [ A, C, T, G]
    The resulting probabilities would be: [A, C, {A: 0.5, T:0.5}, {A:0.5, G:0.5}, A, A]
    
    Parameters
    ----------
    pwm_motif1 : numpy.array
        The position weight matrix of the first motif (ordered from left to right)
    pwm_motif2 : numpy.array
        The position weight matrix of the second motif (ordered from left to right)
    index_of_overlap : int
        The position from start of motif1 to where motif2 starts (pos_motif2-pos_motif1)

    Returns
    -------
    numpy.array
        A position weight matrix of the merged motifs.

    """
    len1 = pwm_motif1.shape[1]
    len2 = pwm_motif1.shape[0]
    overlapped = len1 - (index_of_overlap - 1)  # how much overlapped
    pwm_merged = numpy.zeros((len2, len1 - (index_of_overlap - 1)))
    for i in range(overlapped):
        for z in range(len2):
            p_total = (pwm_motif1.item((z, (i + index_of_overlap - 1))) + pwm_motif2.item((z, i))) / 2
            pwm_merged[z][i] = round(p_total, 5)

    pwm_merged_motifs = numpy.array(
        numpy.concatenate((pwm_motif1[:, :index_of_overlap - 1], pwm_merged, pwm_motif2[:, overlapped:]), axis=1))
    return pwm_merged_motifs


# -------------------------------------- simulate_sequence --------------------------------------------------------


def simulate_sequence(length, pwm_motif, emissions, background_weights, resolve_overlap, dist_fn, total_tries=10,
                      verbose=False, ordered=False):
    """ Simulates DNA with multiple non-overlapping motifs at distributed locations

    Generates DNA of a given length with user-input motifs input with positions governed by
    the user-input distribution. The motif positions are chosen independently with their
    specified distribution over the entire length of the sequence. REJECTION SAMPLING -
    if the motifs overlap, new positions will be sampled from the same distributions.

    Parameters
    ----------
    length : int
        The desired final length of the simulated sequence
    pwm_motif : list(numpy.array)
        A list of numpy.arrays in position weight matrix format for dna ['A', 'C', 'G', 'T']. Can take no motifs with an
        input of an empty list.
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities
    resolve_overlap : {'reject', 'merge'}
        The method for which to resolve overlapping motifs. 'reject' re-samples until dist_fn generates positions
        which don't involve overlapping motifs, 'merge' combines the positional weights of overlapping positions
    dist_fn : list(function)
        A list of functions for each motif that when called outputs a random int position for the motif.
        For an example function: def binom():
                                    return scipy.stats.binom.rvs(...)
    total_tries : int, optional
        Default is 10. The number of attempts at generating a non-overlapping motifs or ordered positions. Preventative
        infinite loop measure. To be used if ordered = True and/or if resolve_overlap = 'reject'.
    verbose : bool, optional
        Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each
        iteration)
    ordered : bool, optional
        Default is `False`. Boolean for rejection sampling motif position generation until the motifs are placed in the
        order of the user's pwm_motif.
    Returns
    -------
    list(str)
        The genomic sequence of length = length as a list of single character strings. If resolve_overlap is 'reject'
        and non-overlapping motifs can not be generated within the provided total_tries, will return an empty string.

    Raises
    ------
    ValueError
        If the position of the motif is generated in the negative or in a position that would make the sequence longer
        than the specified length or the motif positions weren't generated in the order specified (if ordered=True))

    """
    if resolve_overlap == 'reject':

        result = False
        num_tries = 0
        while not result and num_tries < total_tries:

            num_motifs = len(pwm_motif)  # total number of motifs to be embedded

            sample = []  # initialize empty sequence list
            poss_motif = []  # initialize empty list of positions

            for i in range(num_motifs):
                pos_motif = dist_fn[i]()
                # for example
                # def binom():
                #   return scipy.stats.binom.rvs(...)

                poss_motif.append([pos_motif, i, pwm_motif[i]])

                if verbose:
                    print("position of motif {}: ".format(str(i + 1)) + str(poss_motif[i][0]))

            try:
                pos_motif_sorted = sorted(poss_motif, key=operator.itemgetter(
                    0))  # sort based on the pos from [pos of motif, index of motif, pwm of motif]
                if ordered and pos_motif_sorted is not poss_motif:
                    raise (
                        ValueError("motif positions not generated in the order that the user specified.")
                    )

                for ii in range(num_motifs):
                    pos_cur = pos_motif_sorted[ii][0] - len(sample)
                    # subtract len(sample) because of implementation of simulate_background_then_motif
                    index_cur = pos_motif_sorted[ii][1]
                    motif_cur = pos_motif_sorted[ii][2]
                    if len(sample) >= (pos_cur + len(sample)) and len(sample) != 0:
                        raise (
                            ValueError("motif number {} overlapped with motif number {}".format(str(index_cur + 1), str(
                                pos_motif_sorted[ii - 1][1] + 1))))
                    if pos_cur >= length - motif_cur.shape[1]:
                        raise (
                            ValueError("motif number {} generated outside of DNA sequence".format(str(index_cur + 1))))
                    if len(sample) == 0 and pos_cur <= 0:
                        raise (ValueError("motif number {} position generated in the negative"))
                    sample.extend(simulate_background_then_motif(pos_cur, motif_cur, emissions, background_weights))
                result = True  # break out of the while loop
            except ValueError:
                if verbose:
                    print('---------------- try number: ' + str(num_tries + 1) + ' ------------------------')
                num_tries += 1
                pass  # go back into the while loop

        # background sequence after the last motif
        len_sample = len(sample)
        sample.extend(numpy.random.choice(emissions, length - len_sample, p=background_weights))
        if not result:
            print("Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                total_tries))
            sample = []  # delete whatever *wrong* sample was generated
        else:
            print("sequence generated")

        return sample

    elif resolve_overlap == 'merge':

        num_motifs = len(pwm_motif)  # only works if pwm_motifs is inputted as a list

        sample = []  # initialize empty sequence

        poss_motif = []  # initialize empty list for positions

        it_is_not_ordered = True
        num_tries = 0

        while it_is_not_ordered and num_tries < total_tries:
            for i in range(num_motifs):
                pos_motif = dist_fn[i]()  # scipy.stats.rvs(...)
                poss_motif.append([pos_motif, pwm_motif[i], i])

                if verbose:
                    print("position of motif {}: ".format(str(i + 1)) + str(poss_motif[i][0]))

            pos_motif_sorted = sorted(poss_motif,
                                      key=operator.itemgetter(
                                          0))  # sort based on the pos from (pos_motif, index of motif)
            if ordered:
                try:
                    if pos_motif_sorted != poss_motif:
                        num_tries += 1
                        raise (
                            ValueError("motif positions not generated in the order that the user specified.")
                        )
                    it_is_not_ordered = False
                except:
                    pass
            else:
                break
        # the Merging process
        ii = 0  # iterator that takes deletion of list values into accounts

        for _ii in range(num_motifs):
            pos_cur = pos_motif_sorted[ii][0]
            motif_cur = pos_motif_sorted[ii][1]
            index_cur = pos_motif_sorted[ii][2]
            if ii != 0:
                pos_last = pos_motif_sorted[ii - 1][0]
                mot_last = pos_motif_sorted[ii - 1][1]
                index_last = pos_motif_sorted[ii - 1][2]
                if (pos_last + mot_last.shape[1]) >= pos_cur:  # IF THEY ARE OVERLAPPED

                    merged_motif = merge_motif(mot_last, motif_cur, (pos_cur - pos_last + 1))

                    if verbose:
                        print('index_of_overlap: ' + str(pos_cur - pos_last + 1))
                        print('merged_motif: ' + str(merged_motif))
                        print("motif number {} overlapped with motif number {}".format(str(index_cur + 1), str(
                            index_last + 1)))

                    del pos_motif_sorted[ii]
                    pos_motif_sorted[ii - 1][1] = merged_motif  # replace pwm_motif with merged

                    ii -= 1  # don't add to ii because it will be index out of range (we deleted a value in the list)

            if pos_cur >= length - len(motif_cur):
                raise (ValueError("motif number {} generated outside of DNA sequence".format(str(index_cur + 1))))
            if len(sample) == 0 and pos_cur <= 0:
                raise (ValueError("motif number {} position generated in the negative".format(str(index_cur + 1))))
            ii += 1  # update ii because we didn't del pos_motif_sorted[ii]

        # add the motifs one by one from the sorted by position list of [position, pwm, order entered by user]
        for j in range(len(pos_motif_sorted)):
            pos_cur = pos_motif_sorted[j][0] - len(
                sample)  # subtract len(sample) because of how simulate_background_then_motif is used
            motif_cur = numpy.array(pos_motif_sorted[j][1])
            sample.extend(simulate_background_then_motif(pos_cur, motif_cur, emissions, background_weights))

        # background sequence after the last motif
        len_sample = len(sample)

        sample.extend(numpy.random.choice(emissions, length - len_sample, p=background_weights))

        # if the total tries were expended and ordered positions couldn't be generated
        if it_is_not_ordered and ordered:
            sample = []
            print("Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                total_tries))

        return sample


def variable_separation_combine_pwm_motifs(pwm_motif1, pwm_motif2, separation, background_weights):
    """ Combines two pwm motifs into one with background weights of a specific length separating the two

    This can be used by the users before sequence is called if a specific separation between two (or more) motifs is
    desired.

    Parameters
    ----------
    pwm_motif1 : numpy.array
        The position weight matrix of the first motif (ordered from left to right)
    pwm_motif2 : numpy.array
        The position weight matrix of the second motif (ordered from left to right)
    separation : int
        The position of one motif relative to another. This is form the end of the first motif to the start of the second
        motif.
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


class Model(object):
    """  This class holds all of the parameters and allows for sequence simulation
    
    This class will allow for method crossover between different models. That is, it allows for more than just dna simulation.
    It allows for the input of multiple (or no) motifs distributed or non distributed over the length of a simulated sequence.
    
    Parameters
    ----------
    name : str
        A str that describes the title of the model.
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically = ['A', 'C', 'G', 'T'].
    pwm_motifs : list(numpy.array)
        The motifs as a list of numpy.arrays in position weight matrix format for dna ['A', 'C', 'G', 'T']. Can be no
        motifs with an input of an empty list []. Can be embedded in the simulated sequence in the order they are passed
        if ordered = True.
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities.
    dist_fn : list(function)
        A list of functions for each motif that when called outputs a random int position for the motif.
        For an example function: def binom():
                                    return scipy.stats.binom.rvs(...)
    motif_names : list(str), optional
        Default is None. A list of len = len(pwm_motifs) of the names of each motif.



    """

    def __init__(self, name, emissions, pwm_motifs, background_weights, dist_fn, motif_names=None):
        """
        Constructs a `Model` object
        """
        self.emissions = emissions
        self.pwm_motifs = pwm_motifs
        self.name = name
        self.background_weights = background_weights
        self.dist_fn = dist_fn
        self.motif_names = motif_names
        # - - - - - - - - set_params() - - - - - #
        self.length = None
        self.resolve_overlap = None
        self.total_tries_per_sample = 10
        self.ordered = False
        self.params_set = False

    def set_params(self, length, resolve_overlap, total_tries_per_sample=10, ordered=False):
        """ Sets additional parameters to be used when sampling. Is required before .sample() or
        .sample_with_variable_spacing() is called.

        Parameters
        ----------
        length : int
            The desired total length of each sequence
        resolve_overlap : {'reject', 'merge'}
            The method for which to resolve overlapping motifs. 'reject' re-samples until dist_fn generates positions
            which don't involve overlapping motifs, 'merge' combines the positional weights of overlapping positions
        total_tries_per_sample : int, optional
            Default is 10. The number of attempts at generating a non-overlapping motifs or ordered positions.
            Preventative infinite loop measure. To be used if ordered = True and/or if resolve_overlap = 'reject'.
        ordered : bool, optional
            Default is `False`. Boolean for rejection sampling motif position generation until the motifs are placed in
            the order of the user's pwm_motif.

        """
        self.length = length
        self.resolve_overlap = resolve_overlap
        self.total_tries_per_sample = total_tries_per_sample
        self.ordered = ordered
        self.params_set = True

    def show_params(self):
        if not self.params_set:
            raise (ValueError("Model parameters not set with set_params"))
        else:
            print("Length: " + str(self.length) + '\nresolve_overlap: ' + str(
                self.resolve_overlap) + '\ntotal_tries_per_sample: '
                  + str(self.total_tries_per_sample) + '\nordered: ' + str(self.ordered))

    def show_model(self):
        num_motifs = len(self.pwm_motifs)
        print('MODEL NAME:: ' + str(self.name) + '\n---------------------\nEmissions: ' + str(self.emissions))
        if self.motif_names is not None:
            for i in range(num_motifs):
                print('Motif Name: ' + str(self.motif_names[i]) + '\nPosition Distribution: ' + str(self.dist_fn[i]) +
                      'SAMPLED FOUR TIMES: ' + str([self.dist_fn[i]() for ii in range(4)]) + '\nPWM: ' +
                      str(self.pwm_motifs[i]) + '\n- - - - - - - - - - - - ')

        else:
            for i in range(num_motifs):
                print('Position Distribution: ' + str(self.dist_fn[i]) + '\nPWM: ' + str(self.pwm_motifs[i]))

    def sample(self, num_samples, verbose=False):
        """ Simulates DNA with multiple motifs at distributed locations.

        Generates DNA of a given length with user-input motifs input with positions governed by the user-input
        distribution. The motif positions are chosen independently with their specified distribution over the entire
        length of the sequence. To deal with overlapping, the positions are chosen with either rejection sampling or
        merging the overlapping motifs.

        Parameters
        ----------
        num_samples : int
            The desired amount of sequences of the specific model to be output
        verbose : bool
            Debugging tool boolean for whether or not to print motif locations and other information explicitly during
            sampling.

        Returns
        -------
        list(list(str))
            The genomic sequences of length = length as a list of single character strings with motifs embedded at
            specific or distributed locations

        """
        if not self.params_set:
            raise (ValueError("Model parameters not set with set_params"))

        all_sequences = []
        for i in range(num_samples):
            all_sequences.append(
                simulate_sequence(self.length, self.pwm_motifs, self.emissions, self.background_weights,
                                  self.resolve_overlap,
                                  self.dist_fn, total_tries=self.total_tries_per_sample, verbose=verbose,
                                  ordered=self.ordered))

        return all_sequences

    def add_motif(self, index, pwm_motif_added):
        """ Insert a motif into model.pwm_motifs

        Parameters
        ----------
        index
            Where in the list of motifs ( model.pwm_motifs ) should the motif be inserted. Only relevant if
            ordered = True
        pwm_motif_added : numpy.array
            Motif in position weight matrix formatting.
        """
        self.pwm_motifs = self.pwm_motifs.insert(index, pwm_motif_added)

    def merge_motifs(self, index_motif1, index_motif2, index_of_overlap):
        """ Merges two specified motifs from Model.pwm_motifs into one and deletes the second motif

        Parameters
        ----------
        index_motif1 : int
            The index of the first motif to be merged, if order is from left to right.
        index_motif2 : int
            The index of the second motif to be merged, if order is from left to right.
        index_of_overlap : int
            The position from start of motif1 to where motif2 starts (pos_motif2-pos_motif1)
        """
        self.pwm_motifs[index_motif1] = merge_motif(self.pwm_motifs[index_motif1], self.pwm_motifs[index_motif2],
                                                    index_of_overlap)
        del self.pwm_motifs[index_motif2]

    def sample_variable_spacing_motif(self, num_samples, index_motif1, index_motif2, spacing_dist_fn, dist_fn_motif1,
                                      verbose=False):
        """ Simulates DNA with motifs at distributed locations with two motifs separated by a distributed distance.

        This method is the same as .sample() but it allows for the specification of a grouped motif of specific
        separation. The separation distance itself will be passed as a distribution function.

        Parameters
        ----------
        num_samples : int
            Number of desired sequences to be output.
        index_motif1 : int
            The index of the first motif to be combined, if order is from left to right.
        index_motif2 : int
            The index of the second motif to be combined, if order is from left to right.
        spacing_dist_fn : function
            A probability distribution function that when called outputs an int value for the spacing between the motifs.
            It is measured from the end of the first motif to the start of the second motif.
            For example: def binom():
                            return scipy.stats.binom.rvs(n, p):
        dist_fn_motif1 : function
            A probability distribution function that when called outputs an int position for the motif.
        verbose : bool
            Debugging tool boolean for whether or not to print motif locations and other information explicitly during
            sampling.

        Returns
        -------
        list(list(str))
            The genomic sequences of length = length as a list of single character strings with motifs embedded at
            specific or distributed locations
        """
        if not self.params_set:
            raise (ValueError("Model parameters not set with set_params"))

        # generate a combined pwm with spacing described by a user specified distribution
        pwm_motif_spacing_combined = variable_separation_combine_pwm_motifs(self.pwm_motifs[index_motif1],
                                                                            self.pwm_motifs[index_motif2],
                                                                            spacing_dist_fn())
        # replace the motif
        self.pwm_motifs[index_motif1] = pwm_motif_spacing_combined
        # replace the distribution function with user specified ( for clarity's sake )
        self.dist_fn[index_motif1] = dist_fn_motif1
        del self.pwm_motifs[index_motif2]

        return self.sample(num_samples, verbose=verbose)
