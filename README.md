# SeqSpawner

An object-oriented Python library for simulating biological sequences.

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
