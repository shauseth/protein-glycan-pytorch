# SH-I

import numpy as np
import itertools
import torch

monomers = ['Fuc', 'GalNAc', 'Gal', 'GlcNAc', 'GlcA', 'Glc', 'KDN', 'Man', 'Neu5,9Ac2', 'Neu5Ac', 'Neu5Gc']
link_types = ['a', 'b']
occupancies = ['1', '2', '3', '4', '5', '6']
branches = ['[', ']']
sul_phos = ['3S', '4S', '6S', '6P']
linkages = ['Sp0', 'Sp8', 'Sp9', 'Sp10', 'Sp11', 'Sp12', 'Sp13', 'Sp14', 'Sp15', 'Sp16',
            'Sp17', 'Sp18', 'Sp19', 'Sp20', 'Sp21', 'Sp22', 'Sp23', 'Sp24', 'Sp25', 'MDPLys']

def encode(glycan):

    glycan_list = []
    linker_list = []

    for unit in glycan.split('-')[:-1]:

        mono_list = []
        type_list = []
        occu_list = []
        bran_list = []
        supo_list = []

        for monomer in monomers:

            mono_list.append(unit.count(monomer + '('))

        glycan_list.append(mono_list)

        for link_type in link_types:

            type_list.append(unit.split('(')[-1].count(link_type))

        glycan_list.append(type_list)

        for occupancy in occupancies:

            occu_list.append((unit[0] + unit[-1]).count(occupancy))

        glycan_list.append(occu_list)

        for branch in branches:

            bran_list.append(unit.count(branch))

        glycan_list.append(bran_list)

        for supo in sul_phos:

            supo_list.append(unit.count(supo))

        supo_list = supo_list

        glycan_list.append(supo_list)

    for linkage in linkages:

        linker_list.append(glycan.split('-')[-1].count(linkage))

    glycan_vector = np.array(list(itertools.chain.from_iterable(glycan_list)), dtype = float)
    linker_vector = np.array(linker_list, dtype = float)
    zeroes_vector = np.zeros(920 - (len(glycan_vector) + len(linker_vector)), dtype = float)

    vector = np.concatenate((zeroes_vector, glycan_vector, linker_vector))

    assert len(vector) == 920, 'Vector length is not 920 bruh'
    assert np.amax(vector) < 2, 'Vector contains value more than 1 bruh'
    
    return torch.tensor(vector).float()

def vectorize(binding):
    
    tensor = torch.zeros(3)
    
    if binding == 'high':
        tensor[0] = 1.0
    elif binding == 'medium':
        tensor[1] = 1.0
    elif binding == 'low':
        tensor[2] = 1.0
        
    return tensor

def classify(tensor):

    n = tensor.argmax()

    if n == 0:
        binding = 'high'
    elif n == 1:
        binding = 'medium'
    elif n == 2:
        binding = 'low'
        
    return binding