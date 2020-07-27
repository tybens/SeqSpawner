# Tyler Benson. Created July 21, 2020

import abc

import numpy


class Motif(metaclass=abc.ABCMeta):
    """ Abstract base class describing the Motifs to be input into a `MotifList` or into the `Model`

    This abstract base class makes sure that all motifs can be sampled for a sequence and for a pwm. Each `motif` must
    have an alignment that defines where in the `model`'s position value is anchored to.
    """
    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError

    @abc.abstractmethod
    def pwm(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def alignment(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class FixedLengthMotif(Motif, metaclass=abc.ABCMeta):
    """ Abstract clas that inherits `.sample` and `.__str__` and property `alignment`.

    This class introduces `.size` to its subclasses, because a `FixedLengthMotif` will always have a defined size.
    """

    @abc.abstractmethod
    def size(self):
        raise NotImplementedError

    # TODO: additional features may get added


class PWM(FixedLengthMotif):
    """ A class for a fixed length position weight matrix that defines a motif.

    Parameters
    ----------
    pwm : numpy.array([[float]])
        A numpy.array in position weight matrix format for dna. probability of [['A', 'C', 'G', 'T'],...]
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically the order is ['A', 'C', 'G', 'T'].
    alignment : str
        A str { 'left', 'center', 'right' } defining how the `model`'s position value is interpreted. If left, the
        position value is the same, if `model` places the motif so that the middle of the motif is at the position and
        so on..
    """

    def __init__(self, pwm, emissions, alignment):
        """ Constructs a `PWM` object
        """
        self._pwm = pwm
        self._emissions = emissions
        self._alignment = alignment

    def sample(self):
        """
        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`
        """
        seq = []
        for i in range(self._pwm.shape[1]):
            seq.extend(numpy.random.choice(self._emissions, p=self._pwm[:, i]))
        return seq

    def size(self):
        return self._pwm.shape[1]  # If using rows for length and cols for A/C/G/T

    def __str__(self):
        return "PWM: {}\nemissions: {}\nalignment: {}".format(str(self._pwm), str(self._emissions),
                                                              str(self._alignment))

    @property
    def alignment(self):
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        self._alignment = alignment

    def pwm(self):
        """
        Returns
        -------
        numpy.array([[float]])
            returns the position weight matrix that defines the motif. Because it is of fixed length, this pwm will be
            the same every time.
        """
        return self._pwm


class VariableLengthMotif(Motif, metaclass=abc.ABCMeta):
    """ This is to be inherited into another variablePWM class by the user... (?)

    This inheriting class will implement a `.sample` and a property `alignment`
    """
    pass


class VariableLengthPWMMotif(VariableLengthMotif):
    """ Example class that inherits VariableLengthMotif and will have a `.sample` method.

    The input position weight matrix will NOT be implemented this way. This is just a fixed length motif.
    TODO: This will need to be edited and/or implemented by a user


    Parameters
    ----------
    pwm : numpy.array([[float]])
        A numpy.array in position weight matrix format for dna. probability of [['A', 'C', 'G', 'T'],...]
    emissions : list(str)
        A list of the emissions that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically the order is ['A', 'C', 'G', 'T'].
    alignment : str
        A str { 'left', 'center', 'right' } defining how the `model`'s position value is interpreted. If left, the
        position value is the same, if `model` places the motif so that the middle of the motif is at the position and
        so on..
    """

    def __init__(self, pwm, emissions, alignment):
        """ Constructs a `VariableLengthPWMMotif` object
        """
        self._pwm = pwm
        self._emissions = emissions
        self._alignment = alignment

    def sample(self):
        # TODO: to be implemented by the user..?
        pass

    def __str__(self):
        return "{} \nemissions: \nalignment: {}".format(str(self._pwm), str(self._emissions), str(self._alignment))

    @property
    def alignment(self):
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        self._alignment = alignment

    def pwm(self):
        """ This will return the same pwm every time. It is not how a `VariableLengthPWMMotif` should work
        """
        return self._pwm


class VariableRepeatedMotif(VariableLengthMotif):
    """

    Parameters
    ----------
    pwm : object
        A PWM object representing the position weight matrix for the motif and the respective emission possibilities.
    count_dist_fn : function
        A function that outputs an amount of times the motif will repeat

    Raises
    ------
    ValueError
        if the `pwm` argument is not passed as a `PWM` object

    """

    def __init__(self, pwm, count_dist_fn):
        """ Constructs a `VariableRepeatedMotif` object
        """
        # For background, pwm is just an array: [[a, c, t, g]].
        # However, should be able to extend this method to repeated TFBS too.

        if not isinstance(pwm, PWM):
            raise ValueError("PWM object is required to be passed to `pwm`")
        self._pwm = pwm
        self._count_dist_fn = count_dist_fn

    def __str__(self):
        info = ""
        info += self._pwm.__str__()
        info += "\nCounts Distribution Function: {}\n SAMPLED FOUR TIMES: {}\n".format(str(self._count_dist_fn),
                                                                                       str([self._count_dist_fn() for _
                                                                                            in
                                                                                            range(4)]))
        return info

    def pwm(self):
        """
        Returns
        -------
        numpy.array([[float]])
            returns the position weight matrix that defines the motif. Because it is of fixed length, this pwm will be
            the same every time.
        """

        ret = []
        for _ in range(self._count_dist_fn()):
            ret.append(self._pwm.pwm())
        return numpy.hstack(ret)

    def sample(self):
        """
        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`
        """
        ret = []
        for _ in range(self._count_dist_fn()):
            ret.extend(self._pwm.sample())
        return ret

    @property
    def alignment(self):
        return self._pwm.alignment


class FixedRepeatedMotif(FixedLengthMotif):
    """ Class used for motifs that repeat a specific amount of times.

    This is also to be used for background sequences. If this is the case, `PWM.pwm` is just an array:
    the probability of emitting an [['A', 'C', 'T', 'G']]

    Parameters
    ----------
    pwm : object
        A PWM object that represents the pwm and the emission possibilities
    repeats : int
        The amount of times the pwm is repeated during sampling.
    """

    def __init__(self, pwm, repeats):
        # For background, pwm is just an array: [[a, c, t, g]].
        # However, should be able to extend this method to repeated TFBS too.
        if not isinstance(pwm, PWM):
            raise ValueError("PWM object is required to be passed to `pwm`")
        self._pwm = pwm
        self._repeats = repeats

    def __str__(self):
        info = ""
        info += self._pwm.__str__() + "\nRepeats: {}\n Length: {}\n".format(str(self._repeats), str(self.size()))
        return info

    def sample(self):
        ret = []
        for _ in range(self._repeats):
            ret.extend(self._pwm.sample())
        return ret

    def size(self):
        return self._repeats * self._pwm.pwm().shape[1]

    def pwm(self):
        ret = []
        for _ in range(self._repeats):
            ret.append(self._pwm.pwm())
        return numpy.hstack(ret)

    @property
    def alignment(self):
        return self._pwm.alignment


class MotifList(Motif, metaclass=abc.ABCMeta):
    """ Abstract base class for a list of motif objects

    Every `Motiflist` must have an anchor and an alignment.
    """
    @abc.abstractmethod
    def __len__(self):
        """
        Returns
        -------
        int
            The number of motifs in the list
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def anchor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def alignment(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def index_of_anchor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def len_anchored_motif(self):
        raise NotImplementedError


class VariableLengthMotifList(MotifList, VariableLengthMotif):
    """ A class to represent a list of motifs (that can be of of variable length) in a specific order and separation.

    This class is to be input into `model` when the order of motifs and a distribution of the possible separations
    between them is significant. Also, a `Motiflist` object may be passed as motif in the `motifs` list __init__.

    Parameters
    ----------
    motifs : list(object)
        A list of motif objects (i.e. PWM, VariableRepeatedMotif, or a user defined VariableLengthPWMMotif)
    anchor : int
        Index of which motif in the list is the anchor.
    alignment : str
        A str of { 'left', 'center', 'right' } denoting what is the specific anchor location in
        the anchor motif for the entire list of motifs
    """

    def __init__(self, motifs, anchor, alignment):
        # assume that the list of motifs are all motif objects (e.x. PWM )
        # pwm_motifs should be list of `VariableLengthMotif`s and/or `PWM`s
        self._motifs = []
        for x in motifs:
            if not isinstance(x, Motif):
                raise ValueError("Only pass motif objects into `motifs` list")
            self._motifs.append(x)

        self._anchor = anchor
        self._alignment = alignment
        self._index_of_anchor = None
        self._len_anchored_motif = None

    def __len__(self):
        """
        Returns
        -------
        int
            The number of motifs in the list
        """
        return len(self._motifs)

    def __str__(self):
        info = "This is a MotifList object!"
        i = 0
        for x in self._motifs:
            info += "\nmotif index {}\n".format(str(i))
            info += "Sampled example PWM: {}\n".format(str(x.pwm()))
            i += 1

        info += "\nIndex of anchor motif: {}\nAlignment of anchor motif: {}\n".format(str(self._anchor),
                                                                                      str(self.alignment))
        return info

    def pwm(self):
        """

        Returns
        -------
        numpy.array([[float]])
            returns a possible position weight matrix that can define the motif list. Because it is not of f fixed
            length, the pwm is sampled and may be of a different length each time
        """
        pwm_tot = []
        i = 0
        for x in self._motifs:
            if i is self._anchor and i != 0:
                self._index_of_anchor = numpy.hstack(pwm_tot).shape[1]
            elif i is self._anchor and i == 0:
                self._index_of_anchor = 0
            pwm_tot.append(x.pwm())
            if i is self._anchor and i != 0:
                self._len_anchored_motif = numpy.hstack(pwm_tot).shape[1] - self._index_of_anchor
            elif i is self._anchor and i == 0:
                self._len_anchored_motif = pwm_tot[i].shape[1] - self._index_of_anchor
            i += 1

        return numpy.hstack(pwm_tot)

    @property
    def anchor(self):
        """ Anchor getter.
        Returns
        -------
        int
            The index of the anchor motif
        """
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        """ Sets the anchor.
        Parameters
        ----------
        anchor : int
            Index of which motif in the list is the anchor.
        """
        self._anchor = anchor

    @property
    def alignment(self):
        """ Alignment getter.
        Returns
        -------
        str
            Where in the anchor motif is the list anchored. { 'left', 'right', 'center' }

        """
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        """ Sets the alignment
        Parameters
        ----------
        alignment : str
            Where in the anchor motif is the list anchored. { 'left', 'right', 'center' }
        """
        self._alignment = alignment

    @property
    def emissions(self):
        return self._motifs[0].emissions

    def sample(self):
        """
        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`. This samples
            each motif in the list of motifs and appends the list together. Because of the variable length of the list,
            it may not be the same length every call.
        """
        ret = []
        for i in range(len(self)):
            if i is self._anchor:
                self._index_of_anchor = len(ret)
            ret.extend(self._motifs[i].sample())
            if i is self._anchor:
                self._len_anchored_motif = len(ret) - self._index_of_anchor
        return ret

    @property
    def index_of_anchor(self):
        if self._index_of_anchor is None:
            raise ValueError("`.sample` or `.pwm` must be called before `.index_of_anchor`")
        return self._index_of_anchor

    @property
    def len_anchored_motif(self):
        if self._len_anchored_motif is None:
            raise ValueError("`.sample` or `.pwm` must be called before `.len_anchored_motif`")
        return self._len_anchored_motif


class FixedLengthMotifList(MotifList):
    """ A class to represent a list of motifs (of fixed length) in a specific order and separation.

    Similar to VariableLengthMotifLIst, this class is to be input into `model` when the order of motifs and a
    distribution of the possible separation between them is significant.

    Parameters
    ----------
    motifs : list(object)
        A list of 'fixed length' motif objects (i.e. PWM, FixedRepeatedMotif)
    anchor : int
        Index of which motif in the list is the anchor.
    alignment : str
        A str of { 'left', 'center', 'right' } denoting what is the specific anchor location in
        the anchor motif for the entire list of motifs

    """

    def __init__(self, motifs, anchor, alignment):

        # assume that the list of motifs are all motif objects (i.e. PWM )
        self._motifs = []
        for x in motifs:
            if not isinstance(x, Motif):
                raise ValueError("Do not pass emission matrices in `motifs` list")
            if isinstance(x, VariableLengthMotif):
                raise ValueError("A VariableLengthMotif was attempted to be passed into the FixedLengthMotifList")
            self._motifs.append(x)

        self._anchor = anchor  # index for which motif should be considered the center of the motif list (?)
        self._alignment = alignment  # the left right or center of the anchor motif is considered the anchor of the
        # entire list
        self._index_of_anchor = None
        self._len_anchored_motif = None

    @property
    def anchor(self):
        """ Anchor getter.
        Returns
        -------
        int
            The index of the anchor motif
        """
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        """ Anchor setter.
        Parameters
        ----------
        anchor : int
            Index of which motif in the list is the anchor.
        """
        self._anchor = anchor

    @property
    def alignment(self):
        """ Alignment getter.
        Returns
        -------
        str
            Where in the anchor motif is the list anchored. { 'left', 'right', 'center' }

        """
        return self._alignment

    @alignment.setter
    def alignment(self, alignment):
        """ Sets the alignment
        """
        self._alignment = alignment

    def __len__(self):
        """ Number of motifs in the list
        """
        return len(self._motifs)

    def __str__(self):
        info = "This is a MotifList object!"
        i = 0
        for x in self._motifs:
            info += "\nmotif index {}\n".format(str(i))
            info += "exact PWM: {}\n".format(str(x.pwm()))
            i += 1

        info += "\nIndex of anchor motif: {}\nAlignment of anchor motif: {}\n".format(str(self._anchor),
                                                                                      str(self.alignment))
        return info

    def pwm(self):
        """ Sampling a position weight matrix from the given motifs in the list

        Returns
        -------
        numpy.array([[float]])
            returns a position weight matrix that can define the motif list. Because it is of fixed length, the pwm will
            be the same at each call.
        """
        pwm_tot = []
        i = 0
        for x in self._motifs:
            if i is self._anchor and i != 0:
                self._index_of_anchor = numpy.hstack(pwm_tot).shape[1]
            elif i is self._anchor and i == 0:
                self._index_of_anchor = 0
            pwm_tot.append(x.pwm())
            if i is self._anchor and i != 0:
                self._len_anchored_motif = numpy.hstack(pwm_tot).shape[1] - self._index_of_anchor
            elif i is self._anchor and i == 0:
                self._len_anchored_motif = pwm_tot[i].shape[1] - self._index_of_anchor
            i += 1

        return numpy.hstack(pwm_tot)

    def sample(self):
        """ Samples the given motif list and outputs a list of randomly chosen emissions as per the specifications.

        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`. This samples
            each motif in the list of motifs and appends the list together. Because of the fixed length of each motif
            in the list, it will be the same length every call.
        """
        ret = []
        for i in range(len(self)):
            if i is self._anchor:
                self._index_of_anchor = len(ret)
            ret.extend(self._motifs[i].sample())
            if i is self._anchor:
                self._len_anchored_motif = len(ret) - self._index_of_anchor

        return ret

    @property
    def index_of_anchor(self):
        if self._index_of_anchor is None:
            raise ValueError("`.sample` must be called before `.index_of_anchor`")
        return self._index_of_anchor

    @property
    def len_anchored_motif(self):
        if self._len_anchored_motif is None:
            raise ValueError("`.sample` must be called before `.len_anchored_motif`")
        return self._len_anchored_motif

    @property
    def emissions(self):
        return self._motifs[0].emissions


def create_motif_list(motifs, anchor, alignment):
    """ A function that creates either a VariableLengthMotifList or a FixedLengthMotifList

    A `VariableLengthMotifList` is created when one or more of the objects in `motifs` is of variable length and
    `FixedLengthMotifList` is when all the objects are of a specific length. This function is used to create the lists
    of motifs that have distributed separation and are of a specific order.

    Parameters
    ----------
    motifs : list(object)
        A list of motif objects (i.e. PWM, VariableRepeatedMotif, or a user defined VariableLengthPWMMotif)
    anchor : int
        Index of which motif in the list is the anchor.
    alignment : str
        A str of { 'left', 'center', 'right' } denoting what is the specific anchor location in
        the anchor motif for the entire list of motifs

    Returns
    -------
    object
        Either a `VariableLengthMotifList` or a `FixedLengthMotifList`
    """
    for i in range(len(motifs)):
        if isinstance(motifs[i], VariableLengthMotif):
            return VariableLengthMotifList(motifs, anchor, alignment)

    return FixedLengthMotifList(motifs, anchor, alignment)
