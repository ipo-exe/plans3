'''
UFRGS - Universidade Federal do Rio Grande do Sul
IPH - Instituto de Pesquisas Hidráulicas
WARP - Research Group in Water Resources Management and Planning
Porto Alegre, Rio Grande do Sul, Brazil

plans - planning nature-based solutions
Version: 3.0

This software is under the GNU GPL3.0 license

Source code repository: https://github.com/ipo-exe/plans3/
Authors: Ipora Possantti: https://github.com/ipo-exe

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use
Limitations:
    Liability
    Warranty
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license

Module description:
This module stores functions of genetic algortihmns.
'''
import numpy as np
import pandas as pd

# utilitay routines for benchmarking
def get_moea_trivial_solution(show=True):
    import matplotlib.pyplot as plt
    # + sign
    o1_tpl = ((2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (1, 1, 1, 1, 1, 1, 1, 1, 1),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2))
    o1 = np.array(o1_tpl)
    # x sign
    o2_tpl = ((1, 2, 2, 2, 2, 2, 2, 2, 1),
              (2, 1, 2, 2, 2, 2, 2, 1, 2),
              (2, 2, 1, 2, 2, 2, 1, 2, 2),
              (2, 2, 2, 1, 2, 1, 2, 2, 2),
              (2, 2, 2, 2, 1, 2, 2, 2, 2),
              (2, 2, 2, 1, 2, 1, 2, 2, 2),
              (2, 2, 1, 2, 2, 2, 1, 2, 2),
              (2, 1, 2, 2, 2, 2, 2, 1, 2),
              (1, 2, 2, 2, 2, 2, 2, 2, 1))
    # o sign
    o2 = np.array(o2_tpl)
    o3_tpl = ((2, 2, 2, 2, 2, 2, 2, 2, 2),
              (2, 2, 2, 1, 1, 1, 2, 2, 2),
              (2, 2, 1, 2, 2, 2, 1, 2, 2),
              (2, 1, 2, 2, 2, 2, 2, 1, 2),
              (2, 1, 2, 2, 2, 2, 2, 1, 2),
              (2, 1, 2, 2, 2, 2, 2, 1, 2),
              (2, 2, 1, 2, 2, 2, 1, 2, 2),
              (2, 2, 2, 1, 1, 1, 2, 2, 2),
              (2, 2, 2, 2, 2, 2, 2, 2, 2))
    o3 = np.array(o3_tpl)
    # mask
    m0_tpl = ((1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 0, 1, 1),
              (1, 1, 1, 1, 1, 1, 1, 1, 1))
    m0 = np.array(m0_tpl)
    if show:
        plt.imshow(o1)
        plt.show()
        plt.imshow(o2)
        plt.show()
        plt.imshow(o3)
        plt.show()
        plt.imshow(m0)
        plt.show()
    return o1, o2, o3, m0


def get_trivial_solution(show=False):
    """
    Get trivial image solution
    :param show: boolean to control image display
    :return: full solution 2d array and mask 2d array
    """
    import matplotlib.pyplot as plt
    answer = ((1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
              (1, 1, 1, 1, 2, 2, 2, 2, 2, 2),
              (1, 1, 1, 1, 3, 3, 3, 2, 2, 2),
              (1, 1, 1, 1, 3, 3, 3, 2, 2, 2),
              (1, 1, 1, 1, 4, 3, 3, 2, 2, 2),
              (1, 1, 1, 4, 4, 4, 3, 4, 2, 2),
              (1, 1, 4, 4, 5, 4, 4, 4, 4, 4),
              (4, 4, 4, 5, 5, 5, 4, 4, 4, 4),
              (4, 4, 5, 5, 5, 5, 5, 4, 4, 4),
              (4, 4, 4, 4, 4, 4, 4, 4, 4, 4))
    solution_full = np.array(answer)
    if show:
        plt.imshow(solution_full, cmap='viridis')
        plt.show()
    mask = ((1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 1, 1, 1, 1, 0),
            (0, 0, 0, 0, 3, 3, 2, 2, 2, 0),
            (1, 0, 0, 1, 3, 3, 3, 2, 2, 0),
            (0, 0, 1, 1, 4, 3, 3, 2, 2, 0),
            (0, 1, 1, 4, 4, 4, 3, 2, 2, 0),
            (0, 1, 4, 4, 4, 4, 4, 2, 2, 0),
            (0, 4, 4, 5, 5, 5, 4, 4, 4, 0),
            (0, 4, 4, 5, 5, 5, 4, 4, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    mask_array  = (np.array(mask) > 0) * 1
    if show:
        plt.imshow(solution_full * mask_array, cmap='viridis')
        plt.show()
    return solution_full, mask_array


def get_large_solution(seed=666, size=30, show=False):
    """
    Get large image solutions
    :param seed: int seed for random generator
    :param size: int image size (square of size x size)
    :param show: boolean control
    :return: solution 2d array and mask 2d array
    """
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt
    np.random.seed(seed)
    m = np.random.random(size=(size, size))
    m_smoth = gaussian_filter(m, sigma=2)
    p = np.percentile(m_smoth, (80, 40, 60, 90))
    mask = (m_smoth < p[0]) * 1
    #plt.imshow(mask, cmap='Greys_r')
    #plt.show()
    m = np.random.random(size=(size, size))
    #plt.imshow(m, cmap='viridis')
    #plt.show()
    m_smoth = gaussian_filter(m, sigma=3)
    #plt.imshow(m_smoth, cmap='viridis')
    #plt.show()
    p = np.percentile(m_smoth, (10, 40, 60, 90))
    #print(p)
    sol = ((m_smoth < p[0]) * 1) + \
          ((m_smoth >= p[0]) * (m_smoth < p[1]) * 2) + \
          ((m_smoth >= p[1]) * (m_smoth < p[2]) * 3) + \
          ((m_smoth >= p[2]) * (m_smoth < p[3]) * 4) + \
          ((m_smoth >= p[3]) * 5)
    if show:
        plt.imshow(sol, cmap='viridis')
        plt.show()
        plt.imshow(sol * mask, cmap='viridis')
        plt.show()
    return sol, mask

# utilitary routines for plotting:
def plot_trace_generations(evolution, mask, sol, folder='.', step=1):
    import matplotlib.pyplot as plt
    gens_lst = list()
    scores_gens_lst = list()
    for i in range(0, len(evolution), step):
        exp_lst = list()
        # express genes
        for j in range(36):
            # print(np.shape(sol_array)[0])
            lcl_exp = express_2darray_mask(evolution[i]['Parents'][j][0], mask)
            # print(lcl_exp)
            exp_lst.append(lcl_exp[:])
        plot_generation(folder=folder, sol=sol * mask, gen=exp_lst, ids=evolution[i]['Ids'][:36],
                            scores=evolution[i]['Scores'][:36], nm=str(i + 1))
        print('plot ' + str(i + 1))
        scores_gens_lst.append(evolution[i]['Scores'][0])
        # print(evolution[i]['Scores'][0])
        gens_lst.append(i + 1)
    plt.plot(gens_lst, scores_gens_lst, 'k-')
    plt.ylabel('Score')
    plt.xlabel('Generations')
    plt.savefig(folder + '/convegence.png')
    plt.close()


def plot_trace_gen_moea(evolution, mask, sols, folder='.', step=1, cmap='Greys_r'):
    import matplotlib.pyplot as plt
    gens_lst = list()
    scores_gens_lst = list()
    for i in range(0, len(evolution), step):
        exp_lst = list()
        # express genes
        for j in range(36):
            # print(np.shape(sol_array)[0])
            lcl_exp = express_2darray_mask(evolution[i]['Parents'][j][0], mask)
            # print(lcl_exp)
            exp_lst.append(lcl_exp[:])
        plot_generation_moea(folder=folder, sols=(sols[0] * mask, sols[1] * mask, sols[2] * mask),
                             gen=exp_lst, ids=evolution[i]['Ids'][:36], scores=evolution[i]['Scores'][:36],
                             nm=str(i + 1), cmap=cmap)
        print('plot ' + str(i + 1))
        scores_gens_lst.append(evolution[i]['Scores'][0])
        # print(evolution[i]['Scores'][0])
        gens_lst.append(i + 1)
    plt.plot(gens_lst, scores_gens_lst, 'k-')
    plt.ylabel('Score')
    plt.xlabel('Generations')
    plt.savefig(folder + '/convegence.png')
    plt.close()


def plot_convergence(folder, gens, scores, colors, labels, nm='file'):
    import matplotlib.pyplot as plt
    for i in range(len(gens)):
        plt.plot(gens[i], scores[i], c=colors[i], label=labels[i])
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Generations')
    plt.savefig(folder + '/convergence_' + nm + '.png')
    plt.close()


def plot_generation_moea(folder, sols, gen, ids, scores, nm, cmap='viridis'):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig = plt.figure(figsize=(8, 9))
    fig.suptitle('Generation {}'.format(nm), fontsize=12)
    gs = mpl.gridspec.GridSpec(7, 6, wspace=0.2, hspace=0.4, top=0.90, bottom=0.1, left=0.1, right=0.95)
    #
    #
    ind = 0
    for i in range(7):
        for j in range(6):
            if i == 0 and j == 0:
                plt.subplot(gs[i, j])
                plt.title('Solution 1', fontsize=8, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(sols[0], cmap=cmap, vmin=np.min(sols), vmax=np.max(sols))
            elif i == 0 and j == 1:
                plt.subplot(gs[i, j])
                plt.title('Solution 2', fontsize=8, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(sols[1], cmap=cmap, vmin=np.min(sols), vmax=np.max(sols))
            elif i == 0 and j == 2:
                plt.subplot(gs[i, j])
                plt.title('Solution 3', fontsize=8, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(sols[2], cmap=cmap, vmin=np.min(sols), vmax=np.max(sols))
            elif i == 0:
                pass
            else:
                plt.subplot(gs[i, j])
                plt.title(ids[ind] + ' S:' + str(round(scores[ind], 1)), fontsize=6, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(gen[ind], cmap=cmap, vmin=np.min(sols), vmax=np.max(sols))
                ind = ind + 1
    plt.savefig(folder + '/gen-' +  nm + '.png', dpi=300)
    plt.close(fig=fig)


def plot_generation(folder, sol, gen, ids, scores, nm, cmap='viridis'):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig = plt.figure(figsize=(8, 9))
    fig.suptitle('Generation {}'.format(nm), fontsize=12)
    gs = mpl.gridspec.GridSpec(7, 6, wspace=0.2, hspace=0.4, top=0.95, bottom=0.1, left=0.1, right=0.95)
    #
    #
    ind = 0
    for i in range(7):
        for j in range(6):
            if i == 0 and j == 0:
                plt.subplot(gs[i, j])
                plt.title('Solution', fontsize=8, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(sol, cmap=cmap)
            elif i == 0:
                pass
            else:
                plt.subplot(gs[i, j])
                plt.title(ids[ind] + ' S:' + str(round(scores[ind], 1)), fontsize=6, loc='left')
                plt.xticks([])
                plt.yticks([])
                plt.imshow(gen[ind], cmap=cmap, vmin=np.min(sol), vmax=np.max(sol))
                ind = ind + 1
    plt.savefig(folder + '/gen-' +  nm + '.png', dpi=300)
    plt.close(fig=fig)

# general functions:
def express_string(gene, concat=''):
    """
    Express a string-based gene
    :param gene: gene iterable
    :param concat: string concatenator - default is ''
    :return: gene expression (string)
    """
    aux_lst = list()
    for i in range(len(gene)):
        aux_lst.append(str(gene[i]))
    expression = concat.join(aux_lst)
    return expression


def express_intvalue(gene):
    """
    Express a integer-based gene
    :param gene: gene iterable
    :return: integer
    """
    expression = int(gene[0])
    return expression


def express_floatvalue(gene):
    """
    Express a float-based gene
    :param gene: gene iterable
    :return: float
    """
    expression = float(gene[0])
    return expression


def express_2darray(gene, rowlen=3):
    """
    Express a 2d array based gene
    :param gene: gene iterable
    :param rowlen: int of row length
    :return: 2d numpy array
    """
    matrix = list()
    for i in range(0, len(gene), rowlen):
        lcl_row = gene[i: i + rowlen]
        matrix.append(lcl_row[:])
    return np.array(tuple(matrix))


def express_2darray_mask(gene, mask):
    """
    Express a 2d array based gene considering a boolean mask array
    :param gene: gene iterable
    :param mask: pseudo-boolean (1 and 0) 2d array
    :return: 2d numpy array
    """
    matrix = list()
    mask_1d = np.reshape(mask, np.size(mask))
    gene_id = 0
    for i in range(len(mask_1d)):
        if mask_1d[i] == 1:
            matrix.append(gene[gene_id])
            gene_id = gene_id + 1
        else:
            matrix.append(0)
    matrix_2d = np.array(matrix).reshape(np.shape(mask))
    return matrix_2d


def express_1darray(gene):
    """
    Express a 1d array-based gene
    :param gene: gene iterable
    :return: 1d numpy array
    """
    expression = np.array(gene)
    return expression


def encode_string(string='text', concat=''):
    """
    Encode a string-based phenotype
    :param string: string phenotype
    :param concat: string concatenator
    :return: gene 1d tuple
    """
    gene = string.split(concat)
    return tuple(gene)


def encode_1darray(array):
    """
    Encode a 1d array-based phenotype
    :param array: 1d array phenotype
    :return: gene 1d tuple
    """
    return tuple(array)


def encode_2darray(array):
    """
    Encode a 2d array-based phenotype
    :param array: 2d array phenotype
    :return: gene 1d tuple
    """
    aux_lst = list()
    for i in range(len(array)):
        for j in range(len(array[i])):
            aux_lst.append(array[i][j])
    return tuple(aux_lst)


def encode_2darray_mask(array, mask):
    """
    Encode a 2d array-based phenotype considering a boolean mask array
    :param array: 2d array phenotype
    :param mask: 2d array phenotype boolean mask
    :return: gene 1d tuple
    """
    aux_lst = list()
    for i in range(len(array)):
        for j in range(len(array[i])):
            if mask[i][j] > 0:
                aux_lst.append(array[i][j])
    return tuple(aux_lst)


def generate_population(nucleotides, genesizes, popsize=100):
    """
    genesis of a random population
    :param nucleotides: tuple of nucleotides genes (tuple of tuples)
    :param genesizes: tuple of gene sizes (tuple of ints)
    :param popsize: population size (int)
    :return: tuple of random new dnas
    """
    pop_lst = list()
    for i in range(popsize):
        lcl_solution = generate_dna(nucleotides=nucleotides, genesizes=genesizes)
        pop_lst.append(lcl_solution[:])
    return tuple(pop_lst)


def generate_dna(nucleotides, genesizes):
    """
    genesis of a random dna
    :param nucleotides: tuple of nucleotides genes (tuple of tuples - 2d tuple)
    :param genesizes: tuple of gene sizes (tuple of ints)
    :return: tuple of genes (dna)
    """
    #
    def generate_gene(nucleo_set, size=3):
        def_gene = list()
        def_indexes = np.random.randint(0, high=len(nucleo_set), size=size)
        for i in range(len(def_indexes)):
            def_gene.append(nucleo_set[def_indexes[i]])
        return tuple(def_gene)
    #
    def_dna = list()
    for i in range(len(nucleotides)):
        lcl_gene = generate_gene(nucleo_set=nucleotides[i], size=genesizes[i])
        def_dna.append(lcl_gene)
    return tuple(def_dna)


def reproduction(parenta, parentb, nucleotides, mutrate=0.05, puremutrate=0.10, cutfrac=0.2):
    """
    DNA reproduction with crossover and mutation
    :param parenta: Parent A DNA
    :param parentb: Parent B DNA
    :param nucleotides: tuple of nucleotides genes (tuple of tuples)
    :param mutrate: float fraction of mutation rate
    :return: Offspring A DNA and Offspring B DNA (two returns)
    """
    offsp_a = list()
    offsp_b = list()
    # loop in dna genes
    for i in range(len(parenta)):
        # retrieve parent genes
        parent_gene_a = parenta[i]
        parent_gene_b = parentb[i]
        # crossover
        offsp_gene_a, offsp_gene_b = crossover(parent_gene_a, parent_gene_b, cutfrac=cutfrac)
        # mutation
        offsp_gene_a = mutation(offsp_gene_a, nucleotides[i], mutrate=mutrate, puremutrate=puremutrate)
        offsp_gene_b = mutation(offsp_gene_b, nucleotides[i], mutrate=mutrate, puremutrate=puremutrate)
        # appending
        offsp_a.append(offsp_gene_a[:])
        offsp_b.append(offsp_gene_b[:])
    # return offspring dna
    return tuple(offsp_a), tuple(offsp_b)


def crossover(genea, geneb, cutfrac=0.2):
    """
    Crossover of Parent Gene A and Parent Gene B
    :param genea: Gene A tuple
    :param geneb: Gene B tuple
    :param cutfrac: float fraction of cut in gene
    :return: Offspring A and Offspring B genes
    """
    #
    cutsize = 1
    if len(genea) * cutfrac < 1:
        cutsize = 1
    else:
        cutsize = int(len(genea) * cutfrac)
    # random cutpoint inside gene
    cutpoint = np.random.randint(0, len(genea) - cutsize)
    # extract cuts
    cut_a = genea[cutpoint : cutpoint + cutsize]
    cut_b = geneb[cutpoint : cutpoint + cutsize]
    # convert to list
    offs_a = list(genea[:])
    offs_b = list(geneb[:])
    # cross over:
    offs_a[cutpoint : cutpoint + cutsize] = cut_b[:]  # cut B goes in gene A
    offs_b[cutpoint : cutpoint + cutsize] = cut_a[:]  # cut A goes in gene B
    # return tuples
    return tuple(offs_a), tuple(offs_b)


def mutation(gene, gene_nucleo, mutrate=0.05, puremutrate=0.1):
    """
    Mutation of a single gene
    :param gene: tuple of gene
    :param gene_nucleo: tuple of gene nucleotides
    :param mutrate: float fraction of mutation rate
    :param puremutrate: float less than 1  - fraction of pure mutation rate given a mutation event
    :return: mutated gene tuple
    """
    # generate a mutation event
    mutevent = np.random.random(1)
    # mutation rate filter:
    if mutevent <= mutrate:
        #print('Mutation! Type: ', end='\t')
        mutsize = 1 # np.random.randint(1, len(gene))  # get number of mutations
        gene_mut = list(gene)  # convert gene tuple to a list
        for i in range(mutsize):
            # get mutation position in gene
            gene_nucleoid = np.random.randint(0, len(gene))
            muttype = np.random.random()
            # pure random mutation:
            if muttype < puremutrate:
                #print('Pure')
                # get the mutated nucleotide position in the list of nucleotides
                mut_nucleoid = np.random.randint(0, len(gene_nucleo))
            # neighborhood biased mutation:
            else:
                #print('Biased')
                # get neighborhood nucleotides
                if gene_nucleoid == 0:
                    neighbor_nucleos = (gene_mut[gene_nucleoid + 1], gene_mut[gene_nucleoid + 2])
                elif gene_nucleoid == len(gene) - 1:
                    neighbor_nucleos = (gene_mut[gene_nucleoid - 1], gene_mut[gene_nucleoid - 2])
                else:
                    neighbor_nucleos = (gene_mut[gene_nucleoid - 1], gene_mut[gene_nucleoid + 1])
                # get the mutated nucleotide position in the list of biesed nucleotides
                mut_nucleoid = np.random.randint(0, len(neighbor_nucleos))
            # replace nucleotide in gene
            gene_mut[gene_nucleoid] = gene_nucleo[mut_nucleoid]
        return tuple(gene_mut)
    else:
        return gene


def generate_offspring(pop, nucleotides, offsfrac=1, mutrate=0.10, puremutrate=0.10, cutfrac=0.2):
    """
    Genesis of new offspring DNA
    :param pop: tuple of population DNA
    :param nucleotides: tuple of genes nucleotides (2d tuple)
    :param offsfrac: positive float - fraction of offspring related to population size
    :param mutrate: float fraction of mutation rate
    :return: tuple of Offspring DNA
    """
    offsp_lst = list()
    offsize = int(offsfrac * len(pop))
    #print('>>> {}'.format(offsize))
    count = 0
    while True:
        # get random order of mating pool
        parents_ids = np.arange(len(pop))
        np.random.shuffle(parents_ids)
        # loop in mating pool
        for i in range(1, len(pop), 2):
            parent_a = pop[parents_ids[i - 1]]
            parent_b = pop[parents_ids[i]]
            offsp_a, offsp_b = reproduction(parent_a, parent_b, nucleotides=nucleotides, mutrate=mutrate,
                                            puremutrate=puremutrate, cutfrac=cutfrac)
            offsp_lst.append(offsp_a)
            count = count + 1
            offsp_lst.append(offsp_b)
            count = count + 1
            #print(count)
        if count >= offsize:
            break
    return tuple(offsp_lst)


def recruitment(pop, offsp):
    """
    Recruitment of Parents and Offspring
    :param pop: tuple of parents DNA
    :param offsp: tuple of offspring DNA
    :return: tuple of recruited population DNA
    """
    #
    aux_lst = list(pop) + list(offsp)
    return tuple(aux_lst)


def fitness_similarity(dna, solution):
    """
    Benchmark global fitness score of a single DNA based on solution similarity
    :param dna: DNA tuple
    :param solution: Best DNA possible (this may be not be available!!)
    :return: float global fitness score (the higher the better, 100 is perfect)
    """
    fit_lst = list()
    len_lst = list()
    #
    # loop in genes
    for i in range(len(dna)):
        lcl_gene = np.array(dna[i])
        lcl_gene_solution = np.array(solution[i])
        lcl_b = (lcl_gene == lcl_gene_solution) * 1  # pseudo boolean similarity array
        lcl_len = len(lcl_gene_solution)
        lcl_fit = 100 * np.sum(lcl_b)/lcl_len
        #print('{}\t{}\t{}\t{}%'.format(lcl_gene, lcl_gene_solution, lcl_b, lcl_fit))
        fit_lst.append(lcl_fit)
        len_lst.append(lcl_len)
    # global fit is the average of local fitness
    gbl_fit = np.sum(np.array(fit_lst) * np.array(len_lst) / np.sum(len_lst))
    return gbl_fit


def fitness_rmse(dna, solution):
    """
    Benchmark global fitness score of a single DNA based on solution Root Mean Squared Error
    :param dna: DNA tuple
    :param solution: tuple of Solution - Best DNA possible (this may be not be available!!)
    :return: float global fitness score (the higher the better, 0 is perfect)
    """
    from analyst import rmse
    fit_lst = list()
    # loop in genes:
    for i in range(len(dna)):
        lcl_gene = np.array(dna[i])
        lcl_gene_solution = np.array(solution[i])
        lcl_fit = rmse(obs=lcl_gene_solution, sim=lcl_gene) * -1  # negative values of rmse
        fit_lst.append(lcl_fit)
    # global fit is the average of local fitness
    return np.mean(fit_lst)


def fitness_moea(dna, solutions):
    """
    Global fitness score of a single DNA MOEA (convergence to the
    :param dna: DNA tuple
    :param solutions: Best DNA possible solutions (this may be not be available!!)
    :return: float global fitness score
    """
    fit_lst = list()
    #
    # loop in genes
    for i in range(len(dna)):
        lcl_gene = np.array(dna[i])
        gene_fit_lst = list()
        len_lst = list()
        # loop in gene multiples solutions:
        lcl_gene_solutions = np.array(solutions[i])
        for j in range(len(lcl_gene_solutions)):
            # assess multiple solutions:
            lcl_solution = lcl_gene_solutions[j]
            lcl_b = (lcl_gene == lcl_solution) * 1  # boolean of matching nucleotides
            lcl_len = len(lcl_solution)  # number of nucleotides
            lcl_gene_fit = 100 * np.sum(lcl_b)/lcl_len  # give the % of matching nucleotides
            gene_fit_lst.append(lcl_gene_fit)
            len_lst.append(lcl_len)
        # the weighted avg of gene solutions
        gene_gbl_fit = np.sum(np.array(gene_fit_lst) * np.array(len_lst) / np.sum(len_lst))
        fit_lst.append(gene_gbl_fit)
    # global fit is the average of gene global fitness
    gbl_fit = np.sum(np.array(fit_lst))/len(fit_lst)
    return gbl_fit


def evolve(pop0, nucleotides, solution, seed, generations=10, offsfrac=1, mutrate=0.20, puremutrate=0.10,
           cutfrac=0.2, tracefrac=0.3, tracepop=False, fittype='similarity', tui=False):
    """
    Benchmark Evolution of DNAs based on the NSGA-II approach but single-objective
    :param pop0: initial population DNAs
    :param nucleotides: tuple of genes nucleotides (tuple of tuples)
    :param solution: tuple of gene solutions (tuple of objects) --- needed for fitness function
    :param seed: int number for random state
    :param mutrate: float - mutation rate (less than 1)
    :param generations: int - number of generations
    :param puremutrate: float - fraction of pure mutations (less than 1)
    :param cutfrac: float -  fraction of gene cut in cross over (less than 0.5)
    :param tracefrac: float - fraction of traced dnas (less than 1)
    :param fittype: string code for type of fittness function. Available: 'similarity' (default), 'rmse'
    :return: list of traced generations
    """
    from sys import getsizeof
    #
    np.random.seed(seed)
    #
    parents = pop0
    trace = list()
    if tracepop:
        trace_pop = list()
    for g in range(generations):
        if tui:
            print('\n\nGeneration {}\n'.format(g + 1))
        # get offstring
        offspring = generate_offspring(parents, offsfrac=offsfrac, nucleotides=nucleotides, mutrate=mutrate,
                                       puremutrate=puremutrate, cutfrac=cutfrac)
        # recruit new population
        population = recruitment(parents, offspring)
        if tui:
            print('Population: {} KB'.format(getsizeof(population)))
        # fit new population
        ids_lst = list()
        scores_lst = list()
        pop_dct = dict()
        if tracepop:
            dnas_lst = list()
        # loop in individuals
        for i in range(len(population)):
            #
            # get local score and id:
            lcl_dna = population[i]  # local dna
            #
            #
            #
            # Get fitness score:
            if fittype == 'similarity':
                lcl_dna_score = fitness_similarity(lcl_dna, solution=solution)
            elif fittype == 'rmse':
                lcl_dna_score = fitness_rmse(lcl_dna, solution=solution)
            #
            #
            lcl_dna_id = 'G' + str(g + 1) + '-' + str(i)
            #
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            ids_lst.append(lcl_dna_id)
            scores_lst.append(lcl_dna_score)
            if tracepop:
                dnas_lst.append(lcl_dna)
        # trace population
        if tracepop:
            trace_pop.append({'DNAs':dnas_lst[:], 'Ids':ids_lst[:], 'Scores':scores_lst[:]})
        #
        # rank new population (Survival)
        df_population_rank = pd.DataFrame({'Id':ids_lst, 'Score':scores_lst})
        df_population_rank.sort_values(by='Score', ascending=False, inplace=True)
        #
        # Selection of mating pool
        df_parents_rank = df_population_rank.nlargest(len(pop0), columns=['Score'])
        #
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        parents_scores = df_parents_rank['Score'].values  # numpy array of float scores
        #
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        parents = tuple(parents_lst)  # parents DNAs
        #
        # printing
        if tui:
            for i in range(10):
                print('{}'.format(round(parents_scores[i], 3)))
        tr_len = int(len(pop0) * tracefrac)
        #print('>>> {}'.format(tr_len))
        #
        # trace parents
        trace.append({'DNAs':parents[:tr_len],
                      'Ids':parents_ids[:tr_len],
                      'Scores':parents_scores[:tr_len]})
        if tui:
            print('Trace size: {} KB'.format(getsizeof(trace)))
            print('Trace len: {}'.format(len(trace)))
        #if parents_scores[i] > 90:
    #
    # returns
    if tracepop:
        return trace, trace_pop
    else:
        return trace


def evolve_moea(pop0, nucleotides, solutions, seed, mutrate=0.10, generations=10):
    """
    Evolution of DNAs
    :param pop0: initial population DNAs
    :param nucleotides: tuple of genes nucleotides (tuple of tuples)
    :param solutions: 3d Tuple of tuple of gene solutions (tuple of tuple of objects) --- needed for fitness function
    :param seed: int number for random state
    :param mutrate: mutation rate (float fraction)
    :param generations: number of generations
    :return: list of traced generations
    """
    from sys import getsizeof
    #
    np.random.seed(seed)
    #
    parents = pop0
    trace = list()
    for g in range(generations):
        print('\n\nGeneration {}\n'.format(g + 1))
        # get offstring
        offspring = generate_offspring(parents, nucleotides=nucleotides, mutrate=mutrate)
        # recruit new population
        population = recruitment(parents, offspring)
        print('Population: {} KB'.format(getsizeof(population)))
        # fit new population
        ids_lst = list()
        scores_lst = list()
        pop_dct = dict()
        # loop in DNAs
        for i in range(len(population)):
            #
            # get local score and id:
            lcl_dna = population[i]  # local dna
            #
            # Get fitness scores for each objective:
            lcl_dna_score = fitness_moea(lcl_dna, solutions=solutions)
            #lcl_dna_score = fitness(lcl_dna, solution=solution)
            #
            #
            lcl_dna_id = 'G' + str(g + 1) + '-' + str(i)
            #
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            ids_lst.append(lcl_dna_id)
            scores_lst.append(lcl_dna_score)
        #
        # rank new population (Survival)
        df_population_rank = pd.DataFrame({'Id':ids_lst, 'Score':scores_lst})
        df_population_rank.sort_values(by='Score', ascending=False, inplace=True)
        #
        # Selection of mating pool
        df_parents_rank = df_population_rank.nlargest(len(pop0), columns=['Score'])
        #
        parents_ids = df_parents_rank['Id'].values
        parents_scores = df_parents_rank['Score'].values
        #
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        parents = tuple(parents_lst)
        # print
        for i in range(5):
            print('{}%'.format(round(parents_scores[i], 3)))
        tr_len = 40
        trace.append({'Parents':parents[:tr_len],
                      'Ids':parents_ids[:tr_len],
                      'Scores':parents_scores[:tr_len]})
        print('Trace size: {} KB'.format(getsizeof(trace)))
        print('Trace len: {}'.format(len(trace)))
        #if parents_scores[i] > 90:
        #    break
    return trace

# demo routines:
def demo1():
    """
    example of 2D Map GA (single objective)
    :return:
    """
    import matplotlib.pyplot as plt
    # large solution
    sol2, mask2 = get_large_solution(seed=10, size=40, show=True)
    solution = encode_2darray_mask(sol2, mask2)
    nucleo = (1, 2, 3, 4, 5)
    # generate inital population
    popsize = 1000
    pop = generate_population(nucleotides=(nucleo,), genesizes=(len(solution),), popsize=popsize)
    #
    # set parameters
    mutrate = 0.40
    puremutrate = 0.10
    cutfrac = 0.4
    #
    # evolve
    generations = 10
    traced, tracedpop = evolve(pop0=pop, nucleotides=(nucleo,), solution=(solution,), seed=666,
                               generations=generations, mutrate=mutrate, puremutrate=puremutrate,
                               cutfrac=cutfrac, tracefrac=1, tracepop=True)
    # retrieve last one
    last = traced[len(traced) - 1]
    sample_gene = last['DNAs'][0]
    sample_pheno = express_2darray_mask(sample_gene[0], mask2)
    plt.imshow(sample_pheno)
    plt.title('Last generation')
    plt.show()
    #
    # plot generations
    count = 0
    for i in range(generations):
        print(count)
        lcl_scores = np.array(tracedpop[i]['Scores'])
        lcl_scores_parents = np.array(traced[i]['Scores'][:20])
        lcl_y = np.random.random(size=len(lcl_scores))
        lcl_y_parents = np.random.random(size=len(lcl_scores_parents))
        fig = plt.figure(figsize=(4, 6))
        plt.plot(lcl_scores[:], lcl_y[:], 'k.', label='Population')
        plt.plot(lcl_scores_parents, lcl_y_parents, 'ro', label='Fittest')
        plt.yticks([])
        plt.xlim((0, 100))
        plt.ylim((0, 1))
        plt.xlabel('Score (best=100)')
        plt.legend(loc='upper right')
        if count < 10:
            count_id = '00' + str(count)
        elif count >= 10 and count < 100:
            count_id = '0' + str(count)
        else:
            count_id = str(count)
        filename = 'fig_' + count_id
        path = './bin/' + filename + '.png'
        plt.title('Generation ' + count_id)
        plt.savefig(path)
        plt.close()
        count = count + 1
        # plt.show()


def demo2():
    nucleo = tuple(np.arange(0, 101))
    solution = (50, 50, 50, 50, 50, 50, 50)
    # generate inital population
    popsize = 1000
    pop = generate_population(nucleotides=(nucleo,), genesizes=(len(solution),), popsize=popsize)
    #
    # set parameters
    mutrate = 0.40
    puremutrate = 0.10
    cutfrac = 0.4
    #
    # evolve
    generations = 100
    traced, tracedpop = evolve(pop0=pop, nucleotides=(nucleo,), solution=(solution,), seed=666,
                               generations=generations, mutrate=mutrate, puremutrate=puremutrate, cutfrac=cutfrac,
                               tracefrac=1, tracepop=True, fittype='rmse')
    # retrieve last one
    last = traced[len(traced) - 1]
    sample_gene = last['DNAs'][0]
    print(sample_gene)
