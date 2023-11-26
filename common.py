
# Model & training arguments are defined in this file

class Config():
    
    save_ckpt_dir = ''
    result_out_dir = ''
    data_dir = 'MUUFL-20231122T212922Z-001/MUUFL/'
    ckpt_dir = None
    use_gpu = True
    use_dgconv = True
    fix_groups = 1     
    
    num_replicates = 3
    seed = 42
    dataset = 'muufl'
    
    mask_undefined = False
    num_classes = 11
    use_init = True
    
    #------- hyperparams for resnet18 ---------
    model = 'resnet18'
    sample_radius = 5
    epochs = 22
    lr = 0.02
    lr_schedule = [200, 240]
    optimizer = 'sgd'
    batch_size = 48
    
    
    #------ hyperparams for resnet50 ---------
    #model = 'resnet50'
    #sample_radius = 8
    #epochs = 22
    #lr = 0.01
    #lr_schedule = [300, 350]
    #optimizer = 'adam'
    #batch_size = 64
    
    
    #------ hyperparams for tb_cnn ---------
    #model = 'tb_cnn'
    #sample_radius = 5
    #epochs = 22
    #lr = 0.001
    #lr_schedule = None
    #optimizer = 'adam'
    #batch_size = 48
    

