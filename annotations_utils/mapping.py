#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:22:57 2020
"""

########## category mapping ######################
import argparse


def TACO_category_mapping(old_category):
    # our category: cardboard, glass, metal, paper, plastic, trash

    to_glass = ['Glass bottle', 'Broken glass', 'Glass jar']

    to_plastic = ['Clear plastic bottle', 'Other plastic bottle', 'Plastic bottel cap', 'Disposable plastic cup',
                  'Foam cup', 'Other plastic cup', 'Plastic lid', 'Garbage bag', 'Single-use carrier bag',
                  'Polypropylene bag', 'Produce bag', 'Cereal bag', 'Plastic film', 'Crisp packet',
                  'Other plastic wrapper',
                  'Spread tub', 'Tupperware', 'Disposable food container', 'Foam food container',
                  'Other plastic container',
                  'Plastic glooves', 'Plastic utensils', 'Six pack rings', 'Squeezable tube', 'Plastic straw',
                  'Styrofoam piece', 'Other plastic']

    to_metal = ['Scrap metal', 'Pop tab', 'Metal lid', 'Food Can',
                'Drink can', 'Metal bottle cap', 'Aluminium foil']

    to_cardboard = ['Corrugated carton', 'Drink carton', 'Egg carton', 'Meal carton', 'Other carton']

    to_paper = ['Paper cup', 'Magazine paper', 'Tissue', 'Wrapping paper', 'Normal paper',
                'Paper bag', 'Pizza box', 'Paper straw', 'Toilet tube']

    if old_category in to_plastic:
        return 'plastic'
    elif old_category in to_cardboard:
        return 'cardboard'
    elif old_category in to_paper:
        return 'paper'
    elif old_category in to_metal:
        return 'metal'
    elif old_category in to_glass:
        return 'glass'
    else:
        return 'trash'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Taco labels into our labels.')
    parser.add_argument('-C', '--category', help="input the old category", type=str)
    args = parser.parse_args()
    print(TACO_category_mapping(args.category))
