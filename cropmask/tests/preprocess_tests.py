import cropmask.preprocess as pp

def test_make_dirs():
    
    wflow = pp.PreprocessWorkflow()
    
    for i in directory_list:
        assert os.exists(i)
    
    
    
