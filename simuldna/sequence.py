import numpy
import operator


def simulate_sequence_with_single_motif(length, pos_motif, pwm_motif, background_weights):
    """Simulates a dna simuldna with one embedded motif at a specific location.

    Parameters
    ----------
    length : int
        The desired total length of the simuldna
    pos_motif : int
        The index of the location of the motif (not starting at 0)
    pwm_motif : numpy.array
        The motif as a numpy.array in position weight matrix format for dna
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities

    Returns
    -------
    list(str)
        The genomic simuldna of length = length as a list of single character strings with one motif embedded at a
        specific location

    """
    nts = numpy.array(['A', 'C', 'G', 'T'])
    seq = []

    # background before motif
    seq.extend(numpy.random.choice(nts, pos_motif - 1, p=background_weights))

    # motif 
    len_motif = pwm_motif.shape[1]
    for i in range(len_motif):
        seq.extend(numpy.random.choice(nts, p=[pwm_motif[0][i], pwm_motif[1][i], pwm_motif[2][i], pwm_motif[3][i]]))

    # background after
    seq.extend(numpy.random.choice(nts, length - (pos_motif + len_motif - 1), p=background_weights))

    return seq


def simulate_background_then_motif(pos_motif, pwm_motif, background_weights):
    """Simulates DNA of background up until the motif is generated (bg then motif).
    
    Generates a DNA simuldna of a length of posion of the motif + length of the motif - 1
    That is, there is no background simuldna generated after the motif is generated. This
    is to be used in simulate_sequence.
    
    Parameters
    ----------
    pos_motif : int
        The index of the location of the motif (not starting at 0)
    pwm_motif : numpy.array
        The motif as a numpy.array in position weight matrix format for dna ['A', 'C', 'G', 'T']
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities
    Returns
    -------
    list(str)
        A list of single character strings representing DNA nucleotides of the background then the motif.

    """

    nts = numpy.array(['A', 'C', 'G', 'T'])
    seq = []

    # background before motif
    seq.extend(numpy.random.choice(nts, pos_motif - 1, p=background_weights))

    # motif 
    len_motif = pwm_motif.shape[1]
    for i in range(len_motif):
        seq.extend(numpy.random.choice(nts, p=[pwm_motif[0][i], pwm_motif[1][i], pwm_motif[2][i], pwm_motif[3][i]]))

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
        frequency of nucleotides in the background simuldna. ['A', 'C', 'G', 'T']

    """

    c_frac = gc_frac / 2
    bg_weights = [(1 / 2 - c_frac), c_frac, c_frac, (1 / 2 - c_frac)]
    return bg_weights


def simulate_sequence_not_independently(length, pwm_motif, background_weights, dist_fn, verbose=False):
    """ Simulates DNA with multiple motifs at locations distributed relative to each previous motif location.

    The motif locations are sampled using the user-input distribution function; however, the output position from dist_fn
    will be added to the current length of the simuldna. That is, the probability distribution is shifted by the
    previous motif position + the previous motif's length.

    Parameters
    ----------
    length : int
        The desired total length of the simuldna
    dist_fn : list(function)
        A list of distribution functions that when called outputs a random int position for the motif.
        For example: scipy.stats.binom.rvs(...)
    verbose : bool, optional
        Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each
        iteration)
    pwm_motif : list(numpy.array)
        The motifs as a list of numpy.arrays in position weight matrix format for dna ['A', 'C', 'G', 'T']. Can be no
        motifs with an input of an empty list [].
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities

    Returns
    -------
    list(str)
        The genomic simuldna of length = length as a list of single character strings.
    """

    numpy.random.shuffle(pwm_motif)  # shuffle the motifs to be rid of ordering bias
    num_motifs = len(pwm_motif)  # total number of motifs to be embedded
    sample = []  # initialize empty simuldna list

    for i in range(num_motifs):

        pos_motif = dist_fn[i]()  # scipy.stats.rvs(...)

        if verbose:
            print(" position of motif {}: ".format(str(i + 1)) + str(pos_motif + len(sample)))

        if len(sample) >= (pos_motif + len(sample)) or (pos_motif + len(sample)) >= length - pwm_motif[i].shape[1]:
            raise (ValueError("motif number {} generated outside of DNA simuldna".format(str(i + 1))))

        sample.extend(simulate_background_then_motif(pos_motif, pwm_motif[i], background_weights))

    # background simuldna after the last motif
    nts = numpy.array(['A', 'C', 'G', 'T'])
    len_sample = len(sample)
    sample.extend(numpy.random.choice(nts, length - len_sample, p=background_weights))
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
    overlapped = len1 - (index_of_overlap - 1)  # how much overlapped
    pwm_merged = numpy.zeros((4, len1 - (index_of_overlap - 1)))
    for i in range(overlapped):
        for z in range(4):
            p_total = (pwm_motif1.item((z, (i + index_of_overlap - 1))) + pwm_motif2.item((z, i))) / 2
            pwm_merged[z][i] = round(p_total, 5)

    pwm_merged_motifs = numpy.array(
        numpy.concatenate((pwm_motif1[:, :index_of_overlap - 1], pwm_merged, pwm_motif2[:, overlapped:]), axis=1))
    return pwm_merged_motifs


# -------------------------------------- simulate_sequence --------------------------------------------------------


def simulate_sequence(length, pwm_motif, background_weights, resolve_overlap, dist_fn, total_tries=10,
                      verbose=False):
    """ Simulates DNA with multiple non-overlapping motifs at distributed locations

    Generates DNA of a given length with user-input motifs input with positions governed by
    the user-input distribution. The motif positions are chosen independently with their
    specified distribution over the entire length of the simuldna. REJECTION SAMPLING -
    if the motifs overlap, new positions will be sampled from the same distributions.

    Parameters
    ----------
    length : int
        The desired final length of the simulated simuldna
    pwm_motif : list(numpy.array)
        A list of numpy.arrays in position weight matrix format for dna ['A', 'C', 'G', 'T']. Can take no motifs with an
        input of an empty list.
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
        Default is 10. The number of attempts at generating a non-overlapping simuldna. Preventative infinite loop measure.
    verbose : bool, optional
        Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each iteration)

    Returns
    -------
    list(str)
        The genomic simuldna of length = length as a list of single character strings. If resolve_overlap is 'reject'
        and non-overlapping motifs can not be generated within the provided total_tries, will return an empty string.

    Raises
    ------
    ValueError
        If the position of the motif is generated in the negative or in a position that would make the simuldna longer
        than the specified length.

    """
    if resolve_overlap == 'reject':

        result = False
        num_tries = 0
        while not result and num_tries < total_tries:

            num_motifs = len(pwm_motif)  # total number of motifs to be embedded

            sample = []  # initialize empty simuldna list
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
                            ValueError("motif number {} generated outside of DNA simuldna".format(str(index_cur + 1))))
                    if len(sample) == 0 and pos_cur <= 0:
                        raise (ValueError("motif number {} position generated in the negative"))
                    sample.extend(simulate_background_then_motif(pos_cur, motif_cur, background_weights))
                result = True  # break out of the while loop
            except ValueError as ve:
                if verbose:
                    print('---------------- try number: ' + str(num_tries + 1) + ' ------------------------')
                num_tries += 1
                pass  # go back into the while loop

        # background simuldna after the last motif
        nts = numpy.array(['A', 'C', 'G', 'T'])
        len_sample = len(sample)
        sample.extend(numpy.random.choice(nts, length - len_sample, p=background_weights))
        if not result:
            print("Sequence could not be generated without overlap in {} tries".format(total_tries))
            sample = []  # delete whatever *wrong* sample was generated
        else:
            print("simuldna generated")

        return sample

    elif resolve_overlap == 'merge':

        num_motifs = len(pwm_motif)  # only works if pwm_motifs is imputted as a list

        sample = []  # initialize empty simuldna

        poss_motif = []  # initialize empty list for positions

        for i in range(num_motifs):
            pos_motif = dist_fn[i]()  # scipy.stats.rvs(...)
            poss_motif.append([pos_motif, pwm_motif[i], i])

            if verbose:
                print("position of motif {}: ".format(str(i + 1)) + str(poss_motif[i][0]))

        pos_motif_sorted = sorted(poss_motif,
                                  key=operator.itemgetter(0))  # sort based on the pos from (pos_motif, index of motif)

        # the Merging process
        ii = 0  # iterator that takes deletion of list values into accounts

        for _ii in range(num_motifs):
            pos_cur = pos_motif_sorted[ii][0]
            motif_cur = pos_motif_sorted[ii][1]
            index_cur = pos_motif_sorted[ii][2]
            if ii != 0:
                pos_last = pos_motif_sorted[ii - 1][0]
                mot_last = pos_motif_sorted[ii - 1][1]
                index_last = pos_motif_sorted[ii-1][2]
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
                raise (ValueError("motif number {} generated outside of DNA simuldna".format(str(index_cur + 1))))
            if len(sample) == 0 and pos_cur <= 0:
                raise (ValueError("motif number {} position generated in the negative".format(str(index_cur + 1))))
            ii += 1  # update ii because we didn't del pos_motif_sorted[ii]

        # add the motifs one by one from the sorted by position list of [position, pwm, order entered by user]
        for j in range(len(pos_motif_sorted)):
            pos_cur = pos_motif_sorted[j][0] - len(
                sample)  # subtract len(sample) because of how simulate_background_then_motif is used
            motif_cur = numpy.array(pos_motif_sorted[j][1])
            sample.extend(simulate_background_then_motif(pos_cur, motif_cur, background_weights))

        # background simuldna after the last motif
        nts = numpy.array(['A', 'C', 'G', 'T'])
        len_sample = len(sample)

        sample.extend(numpy.random.choice(nts, length - len_sample, p=background_weights))

        return sample
