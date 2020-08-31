# SeqSpawner

An object-oriented Python library for simulating biological sequences.

##### Model(object):
``` This class holds all of the parameters and allows for sequence simulation

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

    
### Examples:
#### Single Motif Embedded
```# load position weight matrix from .meme file
ctcf_pwm = PWM.from_meme_file("/home/tylerbenson/CTCF.meme")  

# function that returns motif location
loc_fn_10 = lambda: 10                         

# background weights 
background = [0.25, 0.25, 0.25, 0.25]

model = Model("test1", 
              pwm.alphabet,
              pwm,
              background,
              loc_fn_10,
              length=50)
model.sample(n=1)
```
#### Two motifs with variable spacing
```tbp_pwm = PWM.from_meme_file("/home/tylerbenson/TBP.meme")
ctcf_pwm = PWM.from_meme_file("/home/tylerbenson/CTCF.meme")

# all adenines separating the variable separation
background_pwm = numpy.array([[1], [0], [0], [0]])  
background = [1, 0, 0, 0] # all adenines

sep_fn = lambda: int(numpy.random.choice(4))+1
repeated_bg = VariableRepeatedMotif(PWM(background_pwm, tbp_pwm.alphabet), sep_fn)
list_test = create_motif_list([ctcf_pwm, repeated_bg, tbp_pwm])
model = Model("test2", 
              list_test.alphabet,
              list_test,
              background,   
              loc_fn_10,
              length=50)
model.sample(num_samples=10)

OUTPUT:
['AAAAAAAAATTAACGCCAGAGGGCGCTCAAAACTATAAATTCCGGGTAAA',
 'AAAAAAAAACGACCACTAGAAGGCACCAAAAACTATATAAAGAGTCGAAA',
 'AAAAAAAAACTGCCACTAGGGGTCGGAGAATATAAATTGGTTGTAAAAAA',
 'AAAAAAAAATTACCACAAGGGGGCGCTAAAAAGCTTAAAAAGCCTGGAAA',
 'AAAAAAAAACCGCCAGTAGGGGGCGGTTAAAACTATTTCTACTCGTGAAA',
 'AAAAAAAAACTGCCTCAAGGTGGCTGTCAAAATACAAAGGTCGGGAAAAA',
 'AAAAAAAAATCACCAGCAGGGGGCGCTAAACTATAAAAATGAAGTAAAAA',
 'AAAAAAAAACCACTTCAAGATGGCGGCCAATCATATAATTGAGCCAAAAA',
 'AAAAAAAAATAGCCGCCAGGGGACACAAAAAGTATATAAAGACTCCAAAA',
 'AAAAAAAAATGTCCACCTGAGGGAGACAAAAATATAAATACGCAAAAAAA']
```
#### List of motif lists - variable spacing of three motifs with a fourth merged if overlapping
```
# Load motifs
tbp_pwm = PWM.from_meme_file("/~/TBP.meme")
ctcf_pwm = PWM.from_meme_file("/~/CTCF.meme")

# create a motif list with lists embedded (list_test and repeated_bg is initialized in above example)
motif_list_of_lists = create_motif_list([tbp_pwm, repeated_bg, list_test])  # variable spacing of pwm and list_test
motifs_all = [motif_list_of_lists, tbp_pwm]  # independent spacing of pwm2 and list_inception
loc_fn_10, loc_fn_20 = lambda:10, lambda: 20          
dist_fns = [loc_fn_10, loc_fn_20]  # list of position functions for each motif
background = [1, 0, 0, 0]
model = Model("test3", 
              motif_list_of_lists.alphabet,
              motifs_all,
              background,
              dist_fns,
              length=70,
              resolve_overlap='merge')  # resolve overlap by merging the motifs
model.sample(1)

OUTPUT:
['AAAAAAAAACTATAAATGGGTAGAAAAGGGCAAACTAGAGGGCGCTAAGCATAAAAGCAATTCAAAAAAA']
```
