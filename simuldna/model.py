# Tyler Benson. Created July 15, 2020.

import operator

import numpy

from simuldna.sequence import simulate_background_then_motif
from simuldna.sequence import variable_separation_combine_pwm_motifs


class Model(object):
    """This class holds all of the parameters and allows for sequence simulation
    
    This class will allow for method crossover between different models. That is, it allows for more than just dna
    simulation. It also allows for the input of multiple (or no) motifs distributed over the length of
    a simulated sequence.
    
    Parameters
    ----------
    name : str
        A str that describes the title of the model.
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically the order is ['A', 'C', 'G', 'T'].
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
        self.variable_spacing_motif_indices = None  # tuples for each grouped motif
        self.variable_spacing_dist_fns = None  # need to delete the self.dist_fn as well as the self.pwm_motifs when
        # combination occurs (and alert the user to this!
        self.params_set = False

    def set_params(self, length, resolve_overlap, total_tries_per_sample=10, variable_spacing_motif_indices=None,
                   variable_spacing_dist_fns=None, ordered=False):
        """ Sets additional parameters to be used when sampling. Is required before `model.sample` or
        is called.

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
        variable_spacing_motif_indices : list(tuple), optional
            Default is None. If variable spacing occurs between two motifs. The indices of such motifs are
            parsed as tuples. If more than just two have spacing specifications, a list of tuples.
        variable_spacing_dist_fns : list(function), optional
            Default is None. A list of length == len(variable_spacing_motif_indices) that express a distribution for the
            separation between the two motifs that have variable spacing. This separation is the end of the first motif
            to the start of the second motif.
        ordered : bool, optional
            Default is `False`. Boolean for rejection sampling motif position generation until the motifs are placed in
            the order of the user's pwm_motif.

        """
        self.length = length
        self.resolve_overlap = resolve_overlap
        self.total_tries_per_sample = total_tries_per_sample
        self.variable_spacing_motif_indices = variable_spacing_motif_indices
        self.variable_spacing_dist_fns = variable_spacing_dist_fns
        self.ordered = ordered
        self.params_set = True

    def __str__(self):
        num_motifs = len(self.pwm_motifs)
        if not self.params_set:
            info = "MODEL NAME: {0}\nemissions: {5}\nbackground_weights: {6}\n----PARAMETERS----\nNOT SET WITH `model.set_params`!\nLength: {1}\nresolve_overlap: {2}\ntotal_tries_per_sample: {3}\nordered: {4}".format(str(self.name),
                str(self.length), str(self.resolve_overlap), str(self.total_tries_per_sample), str(self.ordered), str(self.emissions), str(self.background_weights))
        else:
            info = "MODEL NAME: {0}\nemissions: {5}\nbackground_weights: {6}\n----PARAMETERS----\nLength: {1}\nresolve_overlap: {2}\ntotal_tries_per_sample: {3}\nordered: {4}\n".format(str(self.name),
                str(self.length), str(self.resolve_overlap), str(self.total_tries_per_sample), str(self.ordered), str(self.emissions), str(self.background_weights))
            if self.variable_spacing_motif_indices is not None:
                for i in range(len(self.variable_spacing_motif_indices)):
                    info += "\nVariable Spacing Pair: {0}\nSeparation Distribution: {1}\nSAMPLED FOUR TIMES: {2}".format(
                        str(self.variable_spacing_motif_indices[i]), str(self.variable_spacing_dist_fns[i]),
                        str([self.variable_spacing_dist_fns[i]() for ii in range(4)])) + "\nPosition Distribution to be" \
                                                                                         " deleted: {0}".format(
                        str(self.dist_fn[self.variable_spacing_motif_indices[i][1]])
                    )
                info += "\n\n---- motifs -----"
            if self.motif_names is not None:
                for i in range(num_motifs):
                    info += (
                        '\nMotif Name: {0}\nPosition Distribution: {1}  SAMPLED FOUR TIMES: {2}\nPWM:\n{3}\n- - - - - - - - - - - - '.format(
                            str(self.motif_names[i]), str(self.dist_fn[i]), str([self.dist_fn[i]() for ii in range(4)]),
                            str(self.pwm_motifs[i])))
            else:
                for i in range(num_motifs):
                    info += '\nPosition Distribution: {0}\nPWM:\n{1}'.format(str(self.dist_fn[i]),
                                                                             str(self.pwm_motifs[i]))
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
            The genomic sequence of length == length as a list of single character strings. If resolve_overlap is 'reject'
            and non-overlapping motifs can not be generated within the provided total_tries, will return an empty string.

        Raises
        ------
        ValueError
            If the position of the motif is generated in the negative or in a position that would make the sequence longer
            than the specified length or the motif positions weren't generated in the order specified (if ordered=True)

        """
        length = self.length
        pwm_motif = self.pwm_motifs
        emissions = self.emissions
        background_weights = self.background_weights
        resolve_overlap = self.resolve_overlap
        dist_fn = self.dist_fn
        total_tries = self.total_tries_per_sample
        ordered = self.ordered

        # check for variable spacing specifications and change `pwm_motif` and `dist_fn` accordingly
        if self.variable_spacing_motif_indices is not None:
            for i in range(len(self.variable_spacing_motif_indices)):
                # subtract i because every time it goes through for loop it deletes a pwm_motif and thus the indices
                # shift back an amount i
                index_motif1 = self.variable_spacing_motif_indices[i][0] - i
                index_motif2 = self.variable_spacing_motif_indices[i][1] - i
                spacing_dist_fn = self.variable_spacing_dist_fns[i]
                # generate a combined pwm with spacing described by a user specified distribution
                pwm_motif_spacing_combined = variable_separation_combine_pwm_motifs(self.pwm_motifs[index_motif1],
                                                                                    self.pwm_motifs[index_motif2],
                                                                                    spacing_dist_fn())
                # replace the motif
                pwm_motif[index_motif1] = pwm_motif_spacing_combined
                # delete the second combined motif and distribution function
                del pwm_motif[index_motif2]
                del dist_fn[index_motif2]

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
                        raise ValueError("motif positions not generated in the order that the user specified.")
                    for ii in range(num_motifs):
                        pos_cur = pos_motif_sorted[ii][0] - len(sample)
                        # subtract len(sample) because of implementation of simulate_background_then_motif
                        index_cur = pos_motif_sorted[ii][1]
                        motif_cur = pos_motif_sorted[ii][2]
                        if len(sample) >= (pos_cur + len(sample)) and len(sample) != 0:
                            raise ValueError(
                                "motif number {} overlapped with motif number {}".format(str(index_cur + 1),
                                                                                         str(
                                                                                             pos_motif_sorted[
                                                                                                 ii - 1][
                                                                                                 1] + 1)))
                        if pos_cur >= length - motif_cur.shape[1]:
                            raise ValueError(
                                "motif number {} generated outside of DNA sequence".format(str(index_cur + 1)))
                        if len(sample) == 0 and pos_cur <= 0:
                            raise ValueError("motif number {} position generated in the negative")
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
                raise ValueError(
                    "Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                        str(total_tries)))
            else:
                if verbose:
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
                            raise ValueError("motif positions not generated in the order that the user specified.")
                        it_is_not_ordered = False
                    except ValueError:
                        pass
                else:
                    break  # break out of the while loop

            # the Merging process
            ii = 0  # iterator that takes deletion of list values into accounts

            for j in range(num_motifs):
                pos_cur = pos_motif_sorted[ii][0]
                motif_cur = pos_motif_sorted[ii][1]
                index_cur = pos_motif_sorted[ii][2]
                if ii != 0:
                    pos_last = pos_motif_sorted[ii - 1][0]
                    mot_last = pos_motif_sorted[ii - 1][1]
                    index_last = pos_motif_sorted[ii - 1][2]
                    if (pos_last + mot_last.shape[1]) >= pos_cur:  # IF THEY ARE OVERLAPPED

                        merged_motif = Model._merge_motifs(mot_last, motif_cur, (pos_cur - pos_last + 1))

                        if verbose:
                            print('index_of_overlap: ' + str(pos_cur - pos_last + 1))
                            print('merged_motif: ' + str(merged_motif))
                            print("motif number {} overlapped with motif number {}".format(str(index_cur + 1), str(
                                index_last + 1)))

                        del pos_motif_sorted[ii]
                        pos_motif_sorted[ii - 1][1] = merged_motif  # replace pwm_motif with merged

                        ii -= 1  # don't add to ii because it will be index out of range (we deleted a value in the
                        # list)

                if pos_cur >= length - len(motif_cur):
                    raise ValueError("motif number {} generated outside of DNA sequence".format(str(index_cur + 1)))
                if len(sample) == 0 and pos_cur <= 0:
                    raise ValueError("motif number {} position generated in the negative".format(str(index_cur + 1)))
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
                raise ValueError(
                    "Sequence could not be generated without overlap or not in the ordered specified in {} tries".format(
                        str(total_tries)))

            return sample

    def sample(self, num_samples, verbose=False):
        """ Simulates multiple sequences of DNA with multiple motifs at distributed locations.

        Generates DNA of a given length with user-input motifs input with positions governed by the user-input
        distribution. The motif positions are chosen independently with their specified distribution over the entire
        length of the sequence. To deal with overlapping, the positions are chosen with either rejection sampling or
        merging the overlapping motifs. This function calles `model.sample` `num_samples` times.

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
            The genomic sequences of length == length as a list of single character strings with motifs embedded at
            distributed locations

        """
        if not self.params_set:
            raise ValueError("Model parameters not set with set_params")

        all_sequences = []
        for i in range(num_samples):
            all_sequences.append(Model._sample(verbose=verbose))

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

    @staticmethod  # pycharm told me to do this.
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
        overlapped = len1 - (index_of_overlap - 1)  # how much overlapped
        pwm_merged = numpy.zeros((len2, len1 - (index_of_overlap - 1)))
        for i in range(overlapped):
            for z in range(len2):
                p_total = (pwm_motif1.item((z, (i + index_of_overlap - 1))) + pwm_motif2.item((z, i))) / 2
                pwm_merged[z][i] = round(p_total, 5)

        pwm_merged_motifs = numpy.array(
            numpy.concatenate((pwm_motif1[:, :index_of_overlap - 1], pwm_merged, pwm_motif2[:, overlapped:]), axis=1))
        return pwm_merged_motifs