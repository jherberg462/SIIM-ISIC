from create_params import update_params_dict

params = {
    'batch_size' : 128,
    'img_size' : 299,
    'epochs': 350
}

params = update_params_dict(parms)

params['img_size'] = [params['img_size'] ,params['img_size']]

