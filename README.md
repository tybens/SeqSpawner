# simuldna

Generate simulated DNA strands with specific protein-binding/transcription factor motifs embedded. 

### API:
##### simulate_background_then_motif(pos_motif, pwm_motif, background_weights):
```Simulates DNA of background up until the motif is generated (bg then motif).

Generates a DNA sequence of a length of posion of the motif + length of the motif - 1
That is, there is no background sequence generated after the motif is generated. This
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


```
##### gc_frac_to_background_weights(gc_frac):
```Converts gc_frac to a list of weights

Parameters
----------
gc_frac : float
    fraction of G's and C's in background dna sequencing

Returns
-------
list(float)
    frequency of nucleotides in the background sequence. ['A', 'C', 'G', 'T']

```
##### simulate_sequence_not_independently(length, pwm_motif, background_weights, dist_fn, verbose=False):
``` Simulates DNA with multiple motifs at locations distributed relative to each previous motif location.

The motif locations are sampled using the user-input distribution function; however, the output position from dist_fn
will be added to the current length of the sequence. That is, the probability distribution is shifted by the
previous motif position + the previous motif's length.

Parameters
----------
length : int
    The desired total length of the sequence
dist_fn : function
    A distribution function that when called outputs a random int position for the motif.
    For example: scipy.stats.binom.rvs(...)
verbose : bool, optional
    Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each
    iteration)
pwm_motif : numpy.array
    The motif as a numpy.array in position weight matrix format for dna ['A', 'C', 'G', 'T']
background_weights : list(float)
    A list of weights in float between 0 and 1 representing the background nucleotide probabilities

Returns
-------
list(str)
    The genomic sequence of length = length as a list of single character strings. 

```
##### merge_motif(pwm_motif1, pwm_motif2, index_of_overlap):
``` Merges two overlapping motifs.

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

```
##### simulate_sequence(length, pwm_motif, background_weights, resolve_overlap, dist_fn: object, total_tries=10, verbose=False):
``` Simulates DNA with multiple non-overlapping motifs at distributed locations

Generates DNA of a given length with user-input motifs input with positions governed by
the user-input distribution. The motif positions are chosen independently with their
specified distribution over the entire length of the sequence. REJECTION SAMPLING -
if the motifs overlap, new positions will be sampled from the same distributions.

Parameters
----------
length : int
    The desired final length of the simulated sequence
pwm_motif : numpy.array
    numpy.array in position weight matrix format for dna ['A', 'C', 'G', 'T']
background_weights : list(float)
    A list of weights in float between 0 and 1 representing the background nucleotide probabilities
resolve_overlap : {'reject', 'merge'}
    The method for which to resolve overlapping motifs. 'reject' re-samples until dist_fn generates positions
    which don't involve overlapping motifs, 'merge' combines the positional weights of overlapping positions
dist_fn : function
    function that when called outputs a random int position for the motif. for example: scipy.stats.binom.rvs(...)
total_tries : int, optional
    Default is 10. The number of attempts at generating a non-overlapping sequence. Preventative infinite loop measure.
verbose : bool, optional
    Default is `False`. Print the proceedings of the function (i.e. attempt number and motif positions at each iteration)

Returns
-------
list(str)
    The genomic sequence of length = length as a list of single character strings. If resolve_overlap is 'reject'
    and non-overlapping motifs can not be generated within the provided total_tries, will return an empty string.

Raises
------
ValueError
    If the position of the motif is generated in the negative or in a position that would make the sequence longer
    than the specified length.    
```
##### simulate_sequence_with_single_motif(length, pos_motif, pwm_motif, background_weights):
```Simulates a dna sequence with one embedded motif at a specific location.

Parameters
----------
length : int
    The desired total length of the sequence
pos_motif : int
    The index of the location of the motif (not starting at 0)
pwm_motif : numpy.array
    The motif as a numpy.array in position weight matrix format for dna
background_weights : list(float)
    A list of weights in float between 0 and 1 representing the background nucleotide probabilities

Returns
-------
list(str)
    The genomic sequence of length = length as a list of single character strings with one motif embedded at a
    specific location

```