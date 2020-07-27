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
