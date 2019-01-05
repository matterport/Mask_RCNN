'''
pass here by arguments to script, a .py file that was converted from ipynb via following command-line(example):
jupyter nbconvert --to=python MaskRCNN.ipynb
'''

import sys
import nbformat
from nbformat.v4 import new_code_cell,new_notebook

import codecs

# extract path from argument to script
sourceFile = sys.argv[1]
destFile = sourceFile.split(".")[0]
destFile = destFile + '.ipynb'
print("source is:",sourceFile,"\ndest is:", destFile)

def parsePy(fn):
    """ Generator that parses a .py file exported from a IPython notebook and
extracts code cells (whatever is between occurrences of "In[*]:").
Returns a string containing one or more lines
"""
    with open(fn,"r") as f:
        lines = []
        for l in f:
            l1 = l.strip()
            if l1.startswith('# In[') and l1.endswith(']:') and lines:
                yield "".join(lines)
                lines = []
                continue
            lines.append(l)
        if lines:
            yield "".join(lines)

# Create the code cells by parsing the file in input
cells = []
for c in parsePy(sourceFile):
    cells.append(new_code_cell(source=c))

# This creates a V4 Notebook with the code cells extracted above
nb0 = new_notebook(cells=cells,
                   metadata={'language': 'python',})

with codecs.open(destFile, encoding='utf-8', mode='w') as f:
    nbformat.write(nb0, f, 4)