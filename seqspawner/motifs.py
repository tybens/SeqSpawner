# Tyler Benson. Created July 21, 2020

import abc

import numpy


class Motif(metaclass=abc.ABCMeta):
    """ Abstract base class describing the Motifs to be input into a `MotifList` or into the `Model`

    This abstract base class makes sure that all motifs can be sampled for a sequence and for a pwm.
    """

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError

    @abc.abstractmethod
    def pwm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class FixedLengthMotif(Motif, metaclass=abc.ABCMeta):
    """ Abstract class that inherits `.sample` and `.__str__`.

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
    alphabet : list(str)
        A list of the alphabet that are possible with the model to be simulated. This allows for flexibility away from
        only simulating dna.  For dna typically the order is ['A', 'C', 'G', 'T'].
    """

    def __init__(self, pwm, alphabet):
        """ Constructs a `PWM` object

        Returns
        -------
        object
            A `PWM` object that can be sampled for sequences or for a position weight matrix.
        """
        self._pwm = pwm
        self.alphabet = alphabet

    def sample(self):
        """
        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`
        """
        seq = []
        for i in range(self._pwm.shape[1]):
            p = numpy.array(self._pwm[:, i], dtype=numpy.float64)
            p /= p.sum()
            seq.extend(numpy.random.choice(self.alphabet, p=p))
        return "".join(seq)

    def size(self):
        return self._pwm.shape[1]  # If using rows for length and cols for A/C/G/T

    def __str__(self):
        return "PWM: {}\nalphabet: {}".format(str(self._pwm), str(self.alphabet))

    def pwm(self):
        """
        Returns
        -------
        numpy.array([[float]])
            returns the position weight matrix that defines the motif. Because it is of fixed length, this pwm will be
            the same every time.
        """
        return self._pwm

    @classmethod
    def from_meme_file(cls, input_path):
        freqs = list()
        with open(input_path, "r") as read_file:
            # Read in alphabet.
            for line in read_file:
                if line.startswith("ALPHABET= "):
                    break
            line = line.rstrip()
            line = line.split(" ", 1)[1]
            alpha = list(line)

            # Read in motif.
            for line in read_file:
                if line.startswith("MOTIF"):
                    break
            for line in read_file:
                line = line.rstrip()
                if line.startswith("letter-probability matrix: "):
                    break
            _, _, _, alen, _, w, _, _, _, _ = line.split(" ")
            alen = int(alen)
            w = int(w)
            if alen != len(alpha):
                s = "Found alphabet of {} in meme file\"{}\", but alen was {}".format(alpha, input_path, alen)
                raise ValueError(s)
            m = numpy.zeros((alen, w))

            # Iterate over motif.
            i = 0
            for line in read_file:
                line = line.strip().split()
                line = [float(x) for x in line]
                if len(line) != alen:
                    s = "Found {} columns in line {} in meme file\"{}\", but alphabet length is {}".format(len(line), i,
                                                                                                           input_path,
                                                                                                           alen)
                    raise ValueError(s)
                for j, x in enumerate(line):
                    m[j, i] = x
                i += 1
                if i >= w:
                    break
            if i < w:
                s = "Found {} positions in meme file\"{}\", but width should be {}".format(i, input_path, w)
                raise ValueError(s)

            return PWM(pwm=m, alphabet=alpha)


class VariableLengthMotif(Motif, metaclass=abc.ABCMeta):
    """ This is to be inherited into another variablePWM class by the user... (?)

    This inheriting class will implement a `.sample`
    """
    pass


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

        if not isinstance(pwm, FixedLengthMotif):
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
        return "".join(ret)


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
        ret = ""
        for _ in range(self._repeats):
            ret += "".join(self._pwm.sample())
        return ret

    def size(self):
        return self._repeats * self._pwm.pwm().shape[1]

    def pwm(self):
        ret = []
        for _ in range(self._repeats):
            ret.append(self._pwm.pwm())
        return numpy.hstack(ret)


class MotifList(Motif, metaclass=abc.ABCMeta):
    """ Abstract base class for a list of motif objects

    Every `Motiflist`
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


class VariableLengthMotifList(MotifList, VariableLengthMotif):
    """ A class to represent a list of motifs (that can be of of variable length) in a specific order and separation.

    This class is to be input into `model` when the order of motifs and a distribution of the possible separations
    between them is significant. Also, a `Motiflist` object may be passed as motif in the `motifs` list __init__.

    Parameters
    ----------
    motifs : list(object)
        A list of motif objects (i.e. PWM, VariableRepeatedMotif, or a user defined VariableLengthPWMMotif)
    """

    def __init__(self, motifs):
        # assume that the list of motifs are all motif objects (e.x. PWM )
        # pwm_motifs should be list of `VariableLengthMotif`s and/or `PWM`s
        self._motifs = []
        for x in motifs:
            if not isinstance(x, Motif):
                raise ValueError("Only pass motif objects into `motifs` list")
            self._motifs.append(x)

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
            pwm_tot.append(x.pwm())
            i += 1

        return numpy.hstack(pwm_tot)

    @property
    def alphabet(self):
        return self._motifs[0].alphabet

    def sample(self):
        """
        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`. This samples
            each motif in the list of motifs and appends the list together. Because of the variable length of the list,
            it may not be the same length every call.
        """
        ret = ""
        for i in range(len(self)):
            ret += "".join(self._motifs[i].sample())
        return ret


class FixedLengthMotifList(MotifList):
    """ A class to represent a list of motifs (of fixed length) in a specific order and separation.

    Similar to VariableLengthMotifLIst, this class is to be input into `model` when the order of motifs and a
    distribution of the possible separation between them is significant.

    Parameters
    ----------
    motifs : list(object)
        A list of 'fixed length' motif objects (i.e. PWM, FixedRepeatedMotif)

    """

    def __init__(self, motifs):

        # assume that the list of motifs are all motif objects (i.e. PWM )
        self._motifs = []
        for x in motifs:
            if not isinstance(x, Motif):
                raise ValueError("Do not pass emission matrices in `motifs` list")
            if isinstance(x, VariableLengthMotif):
                raise ValueError("A VariableLengthMotif was attempted to be passed into the FixedLengthMotifList")
            self._motifs.append(x)

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
        for x in self._motifs:
            pwm_tot.append(x.pwm())

        return numpy.hstack(pwm_tot)

    def sample(self):
        """ Samples the given motif list and outputs a list of randomly chosen alphabet as per the specifications.

        Returns
        -------
        list(str)
            A list of nucleotide (or whatever emission is passed in) characters of length == `self.size`. This samples
            each motif in the list of motifs and appends the list together. Because of the fixed length of each motif
            in the list, it will be the same length every call.
        """
        ret = ""
        for i in range(len(self)):
            ret += "".join(self._motifs[i].sample())

        return ret

    @property
    def alphabet(self):
        return self._motifs[0].alphabet


def create_motif_list(motifs):
    """ A function that creates either a VariableLengthMotifList or a FixedLengthMotifList

    A `VariableLengthMotifList` is created when one or more of the objects in `motifs` is of variable length and
    `FixedLengthMotifList` is when all the objects are of a specific length. This function is used to create the lists
    of motifs that have distributed separation and are of a specific order.

    Parameters
    ----------
    motifs : list(object)
        A list of motif objects (i.e. PWM, VariableRepeatedMotif, or a user defined VariableLengthPWMMotif)

    Returns
    -------
    object
        Either a `VariableLengthMotifList` or a `FixedLengthMotifList`
    """
    for i in range(len(motifs)):
        if isinstance(motifs[i], VariableLengthMotif):
            return VariableLengthMotifList(motifs)

    return FixedLengthMotifList(motifs)
