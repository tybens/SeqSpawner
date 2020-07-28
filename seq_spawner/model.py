# Tyler Benson. Created July 15, 2020.

import operator

import numpy

from seq_spawner import motifs
from seq_spawner.sequence import simulate_background_then_motif


class Model(object):
    """This class holds all of the parameters and allows for sequence simulation

    This class will allow for method crossover between different models. That is, it allows for more than just dna
    simulation. It also allows for the input of multiple (or no) motifs distributed over the length of
    a simulated sequence.

    Parameters
    ----------
    name : str
        A str that describes the title of the model.
    list_motifs : list(object)
        THe motifs as a list of motif objects (i.e. PWM, VariableLengthPWMMotif, VariableLengthMotifList,
        FixedLengthMotifList)
    background_weights : list(float)
        A list of weights in float between 0 and 1 representing the background nucleotide probabilities.
    dist_fn : list(function)
        A list of functions for each motif that when called outputs a random int position for the motif.
        For an example function: def binom():
                                    return scipy.stats.binom.rvs(...)
    length : int
        The desired total length of each sequence
    motif_names : list(str), optional
        Default is None. A list of length == len(list_motifs) of the names of each motif.
    resolve_overlap : {'reject', 'merge'}, optional
        Default is 'reject'. The method for which to resolve overlapping motifs. 'reject' re-samples until dist_fn
        generates positions which don't involve overlapping motifs, 'merge' combines the positional weights of
        overlapping positions
    total_tries_per_sample : int, optional
        Default is 10. The number of attempts at generating a non-overlapping motifs or ordered positions.
        Preventative infinite loop measure. To be used if ordered == True and/or if resolve_overlap == 'reject'.
    ordered : bool, optional
        Default is `False`. Boolean for rejection sampling motif position generation until the motifs are placed in
        the order of the user's motif.
    """

    def __init__(self, name, alphabet, list_motifs, background_weights, dist_fn, length, motif_names=None,
                 resolve_overlap='reject', total_tries_per_sample=10, ordered=False):
        """
        Constructs a `Model` object
        """
        if not isinstance(list_motifs, list):
            self.list_motifs = [list_motifs]
        else:
            self.list_motifs = list_motifs

        self.alphabet = alphabet  # this is redundant but it seems easiest
        self.name = name
        self.background_weights = background_weights

        if not isinstance(dist_fn, list):
            self.dist_fn = [dist_fn]
        else:
            self.dist_fn = dist_fn

        if not isinstance(motif_names, list):
            self.motif_names = [motif_names]
        else:
            self.motif_names = motif_names

        self.length = length
        self.resolve_overlap = resolve_overlap
        self.total_tries_per_sample = total_tries_per_sample
        self.ordered = ordered

    def __str__(self):
        num_motifs = len(self.list_motifs)
        info = "MODEL NAME: {0}\nalphabet: {5}\nbackground_weights: {6}\n----PARAMETERS----\nLength: {1}\nresolve_overlap: {2}\ntotal_tries_per_sample: {3}\nordered: {4}\n".format(
            str(self.name),
            str(self.length), str(self.resolve_overlap), str(self.total_tries_per_sample), str(self.ordered),
            str(self.alphabet), str(self.background_weights))
        info += "\n\n---- motifs -----"
        if self.motif_names is not None:
            for i in range(num_motifs):
                info += (
                    '\nMotif Name: {0}\nPosition Distribution: {1}  SAMPLED FOUR TIMES: {2}\n{3}\n- - - - - - - - - - - - '.format(
                        str(self.motif_names[i]), str(self.dist_fn[i]), str([self.dist_fn[i]() for ii in range(4)]),
                        str(self.list_motifs[i].__str__())))
        else:
            for i in range(num_motifs):
                info += '\nPosition Distribution: {0}\n{1}'.format(str(self.dist_fn[i]),
                                                                   str(self.list_motifs[i].__str__()))
        return info

    def _sample(self, verbose=False):
        """ Simulates DNA with multiple non-overlapping motifs at distributed locations

        This helper function for `model.sample` generates one DNA sequence of a given length with user-input
        motifs with positions governed by the user-input distribution. The motif positions
        are chosen independently with their specified distribution over the entire length of the
        sequence.

        Parameters
        ----------
        verbose : bool, optional
            Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each
            iteration)

        Returns
        -------
        list(str)
            The genomic sequence of length == length as a list of single character strings. If resolve_overlap is
            'reject' and non-overlapping motifs can not be generated within the provided total_tries, will return an
            empty string.

        Raises
        ------
        ValueError
            If the position of the motif is generated in the negative or in a position that would make the sequence
            longer than the specified length or (if ordered== True) the motif positions weren't generated in the
            order specified

        """
        length = self.length
        motif_objects = self.list_motifs
        alphabet = self.alphabet
        background_weights = self.background_weights
        resolve_overlap = self.resolve_overlap
        dist_fn = self.dist_fn
        total_tries = self.total_tries_per_sample
        ordered = self.ordered

        if resolve_overlap == 'reject':

            result = False  # becomes True if a sequence is able to be generated without overlap
            num_tries = 0  # initialize iterable tries
            sample = ""  # initialize empty sequence list

            while not result and num_tries < total_tries:

                motif_pwm = [motif_objects[i].pwm() for i in range(len(motif_objects))]

                num_motifs = len(motif_pwm)  # total number of motifs to be embedded

                sample = ""  # empty the sequence list
                poss_motif = []  # initialize empty list of positions

                for i in range(num_motifs):
                    pos_motif = dist_fn[i]()

                    poss_motif.append([pos_motif, i, motif_pwm[i]])

                    if verbose:
                        print(
                            "position of motif {}: {}\n".format(
                                str(i + 1), str(poss_motif[i][0])))

                try:
                    pos_motif_sorted = sorted(poss_motif, key=operator.itemgetter(
                        0))  # sort based on the pos from [pos of motif, index of motif, pwm of motif]
                    if ordered and pos_motif_sorted is not poss_motif:
                        raise ValueError("motif positions not generated in the order that the user specified.")
                    for ii in range(num_motifs):
                        pos_cur = pos_motif_sorted[ii][0] - len(sample)
                        # subtract len(sample) because of implementation of simulate_background_then_motif
                        index_cur = pos_motif_sorted[ii][1]
                        motif_cur_pwm = pos_motif_sorted[ii][2]
                        if pos_cur < 0 and len(sample) != 0:
                            raise ValueError(
                                "motif number {} overlapped with motif number {}".format(str(index_cur + 1),
                                                                                         str(
                                                                                             pos_motif_sorted[
                                                                                                 ii - 1][
                                                                                                 1] + 1)))
                        if pos_cur >= length - motif_cur_pwm.shape[1]:
                            raise ValueError(
                                "motif number {} generated outside of DNA sequence".format(str(index_cur + 1)))
                        if len(sample) == 0 and pos_cur <= 0:
                            raise ValueError("motif number {} position generated in the negative")

                        temp = simulate_background_then_motif(pos_cur, motif_cur_pwm, alphabet,
                                                              background_weights)
                        sample += "".join(temp)
                    result = True  # break out of the while loop
                except ValueError as ve:

                    if verbose:
                        print(ve)
                        print('---------------- try number: ' + str(num_tries + 1) + ' ------------------------')
                    num_tries += 1
                    pass  # go back into the while loop

            # background sequence after the last motif
            len_sample = len(sample)
            sample += "".join(numpy.random.choice(alphabet, length - len_sample, p=background_weights))

            if not result:
                raise ValueError(
                    "Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                        str(total_tries)))
            else:
                if verbose:
                    print("sequence generated")

            return sample

        elif resolve_overlap == 'merge':

            motif_pwm = [motif_objects[i].pwm() for i in range(len(motif_objects))]

            num_motifs = len(motif_pwm)  # only works if motifs is inputted as a list

            sample = ""  # initialize empty sequence

            poss_motif = []  # initialize empty list for positions

            it_is_not_ordered = True
            num_tries = 0

            while it_is_not_ordered and num_tries < total_tries:
                for i in range(num_motifs):
                    pos_motif = dist_fn[i]()

                    poss_motif.append([pos_motif, motif_pwm[i], i])

                    if verbose:
                        print("position of motif {}: {}\n".format(str(i + 1), str(poss_motif[i][0])))

                pos_motif_sorted = sorted(poss_motif,
                                          key=operator.itemgetter(
                                              0))  # sort based on the pos from (pos_motif, index of motif)
                if ordered:
                    try:
                        if pos_motif_sorted != poss_motif:
                            num_tries += 1
                            raise ValueError("motif positions not generated in the order that the user specified.")
                        it_is_not_ordered = False
                    except ValueError:
                        pass
                else:
                    break  # break out of the while loop

            # if the total tries were expended and ordered positions couldn't be generated
            if it_is_not_ordered and ordered:
                raise ValueError(
                    "Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                        str(total_tries)))

            # the Merging process
            ii = 0  # iterator that takes deletion of list values into accounts

            for _ in range(num_motifs):
                pos_cur = pos_motif_sorted[ii][0]
                motif_cur_pwm = pos_motif_sorted[ii][1]
                index_cur = pos_motif_sorted[ii][2]
                if ii != 0:
                    pos_last = pos_motif_sorted[ii - 1][0]
                    mot_last_pwm = pos_motif_sorted[ii - 1][1]
                    index_last = pos_motif_sorted[ii - 1][2]
                    if (pos_last + mot_last_pwm.shape[1]) > pos_cur:  # IF THEY ARE OVERLAPPED
                        index_of_overlap = (pos_cur - pos_last + 1)
                        merged_motif = Model._merge_motifs(mot_last_pwm, motif_cur_pwm, index_of_overlap)

                        if verbose:
                            print('index_of_overlap: ' + str(pos_cur - pos_last + 1))
                            print('merged_motif: ' + str(merged_motif))
                            print("motif number {} overlapped with motif number {}".format(str(index_cur + 1), str(
                                index_last + 1)))

                        del pos_motif_sorted[ii]
                        pos_motif_sorted[ii - 1][1] = merged_motif

                        ii -= 1  # don't add to ii because it will be index out of range (we deleted a value in the
                        # list)

                if pos_cur >= length - len(
                        motif_cur_pwm):
                    raise ValueError("motif number {} generated outside of DNA sequence".format(str(index_cur + 1)))
                if len(sample) == 0 and pos_cur <= 0:
                    raise ValueError("motif number {} position generated in the negative".format(str(index_cur + 1)))
                ii += 1  # update ii because we didn't del pos_motif_sorted[ii]

            # add the list_motifs one by one from the sorted by position list of [position, pwm, order entered by user, motif object]
            for j in range(len(pos_motif_sorted)):
                pos_cur = pos_motif_sorted[j][0] - len(
                    sample)  # subtract len(sample) because of how simulate_background_then_motif is used
                motif_cur_pwm = pos_motif_sorted[j][1]
                sample += "".join(
                    simulate_background_then_motif(pos_cur, motif_cur_pwm, alphabet, background_weights))

            # background sequence after the last motif
            len_sample = len(sample)

            try:
                bg_after_last_motif = numpy.random.choice(alphabet, length - len_sample, p=background_weights)
                sample += "".join(bg_after_last_motif)
            except ValueError:
                raise ValueError("Motif generated is too long for input sequence length")

            return sample

    def sample(self, num_samples, verbose=False):
        """ Simulates multiple sequences of DNA with multiple motifs at distributed locations.

        Generates DNA of a given length with user-input motifs input with positions governed by the user-input
        distribution. The motif positions are chosen independently with their specified distribution over the entire
        length of the sequence. To deal with overlapping, the positions are chosen with either rejection sampling or
        merging the overlapping motifs. This function calls `model.sample` `num_samples` times.

        Parameters
        ----------
        num_samples : int
            The desired amount of sequences of the specific model to be output
        verbose : bool, optional
            Default is False. Debugging tool boolean for whether or not to print motif locations and other information
            explicitly during sampling.

        Returns
        -------
        list(list(str))
            The genomic sequences of length == length as a list of single character strings with motifs embedded at
            distributed locations

        """

        all_sequences = []
        for i in range(num_samples):
            all_sequences.append(self._sample(verbose=verbose))

        return all_sequences

    def add_motif(self, index, motif_added):
        """ Insert a motif into model.motifs

        Parameters
        ----------
        index
            Where in the list of motifs `model.list_motifs` should the motif be inserted. Only relevant if
            ordered = True
        motif_added : numpy.array
            Motif in position weight matrix formatting.
        """
        self.list_motifs = self.list_motifs.insert(index, motif_added)

    @staticmethod
    def _merge_motifs(pwm_motif1, pwm_motif2, index_of_overlap):
        """ Helper function for `model._sample` that merges two overlapping motifs.`

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
            A position weight matrix of the merged motif.

        """
        len1 = pwm_motif1.shape[1]
        len2 = pwm_motif1.shape[0]
        len_pwm2 = pwm_motif2.shape[1]
        overlapped = len1 - (index_of_overlap - 1)  # how much overlapped
        encompassed = False

        if overlapped > len_pwm2:  # this adds functionality for if the first motif encompasses the second motif
            overlapped = len_pwm2
            encompassed = True

        pwm_merged = numpy.zeros((len2, overlapped))
        for i in range(overlapped):
            for z in range(len2):
                p_total = (pwm_motif1.item((z, (i + index_of_overlap - 1))) + pwm_motif2.item((z, i))) / 2
                pwm_merged[z][i] = round(p_total, 5)

        if encompassed:
            pwm_merged_motifs = numpy.array(
                numpy.concatenate(
                    (pwm_motif1[:, :index_of_overlap - 1], pwm_merged, pwm_motif1[:, index_of_overlap + len_pwm2 - 1:]),
                    axis=1))
        else:
            pwm_merged_motifs = numpy.array(
                numpy.concatenate((pwm_motif1[:, :index_of_overlap - 1], pwm_merged, pwm_motif2[:, overlapped:]),
                                  axis=1))

        return pwm_merged_motifs
