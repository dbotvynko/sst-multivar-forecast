from omegaconf import OmegaConf
import os

from contrib.ose_pipeline.rec_utils import reconstruct_from_config

class AllLeadtimesReconstructed(Exception):
    def __init__(self, *args):
        super().__init__(*args)

def get_leadtime_start(
        overwrite,
        rec_paths,
        dT,
        
):
    leadtime_start = 0
    max_dT = dT-8
    if not overwrite:
        for leadtime in range(dT//2, max_dT):
            if os.path.exists(rec_paths.format(leadtime)):
                if leadtime == max_dT - 1:
                    raise AllLeadtimesReconstructed('all leadtimes already reconstructed')
                leadtime_start = leadtime + 1 - dT//2

    return leadtime_start

def setup_model_config(
        model_config_path,
        gridded_input_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_', value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SLA')
                     # for SSH inference : value='contrib.data_loading.data.load_ose_data_with_tgt_mask')
                     # value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SLA')  # for SLA training only !
                     #load_ose_data_with_tgt_mask')
    OmegaConf.update(config, key='datamodule.input_da.path', value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path', value='${paths.glorys12_data}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_not_glorys', value='${paths.ref_data_l4}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_l3_data', value='${paths.ose_data_paths}')
    OmegaConf.update(config, key='datamodule.input_da.variable', value='${var_name}')
    OmegaConf.update(config, key='datamodule.input_da.year', value='${year_ose}')
    #if(len(${paths.ref_data_l4}))

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_', value=[min_time, min_time_offseted])
    
    OmegaConf.update(config, key='datamodule.domains.test.time._args_', value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    # LEADTIME OUTPUTS:
    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT = dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config



def setup_model_config_SST(
        model_config_path,
        gridded_input_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite
):
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_', value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SST')
                     # for SSH inference : value='contrib.data_loading.data.load_ose_data_with_tgt_mask')
                     # value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SLA')  # for SLA training only !
                     #load_ose_data_with_tgt_mask')
    OmegaConf.update(config, key='datamodule.input_da.path', value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path', value='${paths.glorys12_data}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_', value=[min_time, min_time_offseted])

    OmegaConf.update(config, key='datamodule.domains.test.time._args_', value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    # LEADTIME OUTPUTS:
    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT = dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config


def setup_model_config_L4(
        model_config_path,
        gridded_input_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite
):
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_', value='contrib.data_loading.data.load_ose_data_with_tgt_mask_L4')
    OmegaConf.update(config, key='datamodule.input_da.path', value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path', value='${paths.glorys12_data}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_', value=[min_time, min_time_offseted])

    OmegaConf.update(config, key='datamodule.domains.test.time._args_', value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    # LEADTIME OUTPUTS:
    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT = dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config


def setup_model_config_OceanFM_SLA(
        model_config_path,
        gridded_input_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    """Like setup_model_config but uses load_ose_data_sla_oceanfm — no SST fields needed."""
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_',
                     value='contrib.data_loading.data.load_ose_data_sla_oceanfm')
    OmegaConf.update(config, key='datamodule.input_da.path',
                     value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path',
                     value='${paths.glorys12_data}')
    OmegaConf.update(config, key='datamodule.input_da.domain',
                     value='${domain.train}')
    OmegaConf.update(config, key='datamodule.input_da.variable',
                     value='${var_name}')
    OmegaConf.update(config, key='datamodule.input_da.year',
                     value='${year_ose}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.test.time._args_', value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT=dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config


def execute_rec_pipeline(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    
    print('-'*60+'\n'+'-'*60+'\nRECONSTRUCTION PIPELINE START:\n')

    print('setting up model config')
    # Detect OceanFM models (FlowMatchingOSSEForecastPatchGPU_SLA_OceanFM) and use dedicated setup
    _raw_cfg = OmegaConf.load(model_config_path)
    _model_target = OmegaConf.select(_raw_cfg, 'model._target_', default='')
    _is_oceanfm = 'OceanFM' in str(_model_target) or 'FlowMatching' in str(_model_target)

    try:
        if _is_oceanfm:
            print('OceanFM model detected — using OceanFM SLA setup')
            config = setup_model_config_OceanFM_SLA(
                model_config_path,
                gridded_input_path,
                rec_paths,
                min_time,
                max_time,
                min_time_offseted,
                max_time_offseted,
                overwrite,
            )
        else:
            config = setup_model_config(
                model_config_path,
                gridded_input_path,
                rec_paths,
                min_time,
                max_time,
                min_time_offseted,
                max_time_offseted,
                overwrite,
            )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n'+'-'*60)
        return

    print('done\n'+'-'*60)

    print('ose reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n'+'-'*60)

    print('RECONSTRUCTION PIPELINE END:\n'+'-'*60+'\n'+'-'*60)


def setup_model_config_SST_SLA_INOUT(
        model_config_path,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    """
    Set up the model config for OSE inference with SST+SLA inputs and
    SST+SLA outputs.  Overrides the datamodule.input_da block to use
    load_ose_data_with_tgt_mask_SST_SLA_INOUT.
    """
    config = OmegaConf.load(model_config_path)

    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(
        config,
        key='datamodule.input_da._target_',
        value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SST_SLA_INOUT',
    )
    OmegaConf.update(config, key='datamodule.input_da.path',                value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path',             value='${paths.glorys12_data}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_not_glorys',  value='${paths.ref_data_l4}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_l3_data',     value='${paths.ose_data_paths}')
    OmegaConf.update(config, key='datamodule.input_da.sla_input_path',       value=sla_input_path)
    OmegaConf.update(config, key='datamodule.input_da.tgt_sla_path',         value=tgt_sla_path)
    OmegaConf.update(config, key='datamodule.input_da.variable',             value='${var_name}')
    OmegaConf.update(config, key='datamodule.input_da.year',                 value='${year_ose}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_',   value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.test.time._args_',  value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    leadtime_start = get_leadtime_start(
        overwrite,
        rec_paths,
        dT=dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)

    return config


def execute_rec_pipeline_SST_SLA_INOUT(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    print('-' * 60 + '\n' + '-' * 60 + '\nRECONSTRUCTION PIPELINE (SST+SLA INOUT) START:\n')

    print('setting up model config')
    try:
        config = setup_model_config_SST_SLA_INOUT(
            model_config_path=model_config_path,
            gridded_input_path=gridded_input_path,
            sla_input_path=sla_input_path,
            tgt_sla_path=tgt_sla_path,
            rec_paths=rec_paths,
            min_time=min_time,
            max_time=max_time,
            min_time_offseted=min_time_offseted,
            max_time_offseted=max_time_offseted,
            overwrite=overwrite,
        )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n' + '-' * 60)
        return

    print('done\n' + '-' * 60)

    print('SST+SLA INOUT reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n' + '-' * 60)


def setup_model_config_SST_SLA_INOUT_with_inputs(
        model_config_path,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
        leadtime_start_override=-1,
):
    """
    Same as setup_model_config_SST_SLA_INOUT but forces
    output_leadtime_start to include j-1 (leadtime -1).
    """
    config = setup_model_config_SST_SLA_INOUT(
        model_config_path=model_config_path,
        gridded_input_path=gridded_input_path,
        sla_input_path=sla_input_path,
        tgt_sla_path=tgt_sla_path,
        rec_paths=rec_paths,
        min_time=min_time,
        max_time=max_time,
        min_time_offseted=min_time_offseted,
        max_time_offseted=max_time_offseted,
        overwrite=overwrite,
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start_override)
    return config


def execute_rec_pipeline_SST_SLA_INOUT_with_inputs(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    print('-' * 60 + '\n' + '-' * 60 + '\nRECONSTRUCTION PIPELINE (SST+SLA INOUT WITH INPUTS) START:\n')

    print('setting up model config')
    try:
        config = setup_model_config_SST_SLA_INOUT_with_inputs(
            model_config_path=model_config_path,
            gridded_input_path=gridded_input_path,
            sla_input_path=sla_input_path,
            tgt_sla_path=tgt_sla_path,
            rec_paths=rec_paths,
            min_time=min_time,
            max_time=max_time,
            min_time_offseted=min_time_offseted,
            max_time_offseted=max_time_offseted,
            overwrite=overwrite,
        )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n' + '-' * 60)
        return

    print('done\n' + '-' * 60)

    print('SST+SLA INOUT with inputs reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n' + '-' * 60)

    print('RECONSTRUCTION PIPELINE (SST+SLA INOUT) END:\n' + '-' * 60 + '\n' + '-' * 60)


def execute_rec_pipeline_L4(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite
):

    print('-'*60+'\n'+'-'*60+'\nRECONSTRUCTION PIPELINE START:\n')

    print('setting up model config')
    try:
        config = setup_model_config_L4(
            model_config_path,
            gridded_input_path,
            rec_paths,
            min_time,
            max_time,
            min_time_offseted,
            max_time_offseted,
            overwrite
        )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n'+'-'*60)
        return

    print('done\n'+'-'*60)

    print('ose reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n'+'-'*60)

    print('RECONSTRUCTION PIPELINE END:\n'+'-'*60+'\n'+'-'*60)


def setup_model_config_SST_SLA_WIND_INOUT(
        model_config_path,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        wind_path,
        rec_paths,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    config = OmegaConf.load(model_config_path)
    OmegaConf.update(config, key='paths.ose_gridded_input_path', value=gridded_input_path)

    del config['datamodule']['input_da']

    OmegaConf.update(config, key='datamodule.input_da._target_',
                     value='contrib.data_loading.data.load_ose_data_with_tgt_mask_SST_SLA_WIND_INOUT')
    OmegaConf.update(config, key='datamodule.input_da.path',                value='${paths.ose_gridded_input_path}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path',             value='${paths.glorys12_data}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_not_glorys',  value='${paths.ref_data_l4}')
    OmegaConf.update(config, key='datamodule.input_da.tgt_path_l3_data',     value='${paths.ose_data_paths}')
    OmegaConf.update(config, key='datamodule.input_da.sla_input_path',       value=sla_input_path)
    OmegaConf.update(config, key='datamodule.input_da.tgt_sla_path',         value=tgt_sla_path)
    OmegaConf.update(config, key='datamodule.input_da.wind_path',            value=wind_path)
    OmegaConf.update(config, key='datamodule.input_da.variable',             value='${var_name}')
    OmegaConf.update(config, key='datamodule.input_da.year',                 value='${year_ose}')

    OmegaConf.update(config, key='datamodule.domains.train.time._args_', value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.val.time._args_',   value=[min_time, min_time_offseted])
    OmegaConf.update(config, key='datamodule.domains.test.time._args_',  value=[min_time, max_time])

    OmegaConf.update(config, key='model.pre_metric_fn.time._args_', value=[min_time_offseted, max_time_offseted])

    leadtime_start = get_leadtime_start(
        overwrite, rec_paths,
        dT=dict(config)['datamodule']['xrds_kw']['patch_dims']['time'],
    )
    OmegaConf.update(config, key='model.output_leadtime_start', value=leadtime_start)
    return config


def execute_rec_pipeline_SST_SLA_WIND_INOUT(
        model_config_path,
        model_ckpt_path,
        rec_path,
        rec_paths,
        xp_name,
        data_name,
        gridded_input_path,
        sla_input_path,
        tgt_sla_path,
        wind_path,
        min_time,
        max_time,
        min_time_offseted,
        max_time_offseted,
        overwrite,
):
    print('-' * 60 + '\nRECONSTRUCTION PIPELINE (SST+SLA+WIND INOUT) START:\n')

    print('setting up model config')
    try:
        config = setup_model_config_SST_SLA_WIND_INOUT(
            model_config_path=model_config_path,
            gridded_input_path=gridded_input_path,
            sla_input_path=sla_input_path,
            tgt_sla_path=tgt_sla_path,
            wind_path=wind_path,
            rec_paths=rec_paths,
            min_time=min_time,
            max_time=max_time,
            min_time_offseted=min_time_offseted,
            max_time_offseted=max_time_offseted,
            overwrite=overwrite,
        )
    except AllLeadtimesReconstructed:
        print('all leadtimes already reconstructed\n' + '-' * 60)
        return

    print('done\n' + '-' * 60)

    print('SST+SLA+WIND INOUT reconstruction starting')
    reconstruct_from_config(config, rec_path, xp_name, data_name, model_ckpt_path)
    print('done\n' + '-' * 60)

    print('RECONSTRUCTION PIPELINE (SST+SLA+WIND INOUT) END:\n' + '-' * 60)

