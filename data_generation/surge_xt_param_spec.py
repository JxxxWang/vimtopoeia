from data_generation.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

SURGE_SIMPLE_PARAM_SPEC = ParamSpec(
    [
        #Amplitude ADSR envelope
        ContinuousParameter(name="a_amp_eg_attack", min=0.0, max=0.1, distribution="log"),  
        ContinuousParameter(name="a_amp_eg_decay", min=0.2, max=0.77, distribution="log"),  # min increased to ensure audibility
        ContinuousParameter(name="a_amp_eg_sustain", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_amp_eg_release", min=0.0, max=0.77
        ),  # max around 4s
        
        CategoricalParameter(
            name="a_amp_eg_envelope_mode",
            values=["Digital"],
            raw_values=[0.25],
            encoding="onehot",
        ),

        # filter 1 parameters
        # ContinuousParameter(
        #     name="a_filter_balance",
        #     min=0.0,
        #     max=1.0,
        #     constant_val_p=0.0,
        #     constant_val=0.0,
        # ),

        CategoricalParameter(
            name="a_filter_1_type",
            values=["LP 12 dB"],
            raw_values=[0.035500000000000004], # Derived from probing (Index 0)
            encoding="onehot",
        ),
        ContinuousParameter(name="a_filter_1_cutoff", min=0.2, max=1.0, distribution="log"), # min increased (approx 100hz)
        ContinuousParameter(name="a_filter_1_feg_mod_amount", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_1_resonance", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_filter_eg_attack", min=0.0, max=0.1
        ),  
        ContinuousParameter(
            name="a_filter_eg_decay", min=0.2, max=0.77 # min increased
        ),  # max around 4s
        ContinuousParameter(
            name="a_filter_eg_release", min=0.0, max=0.77
        ),  # max around 4s
        ContinuousParameter(name="a_filter_eg_sustain", min=0.0, max=1.0),
        CategoricalParameter(
            name="a_filter_eg_envelope_mode",
            values=["Digital"],
            raw_values=[0.25],
            encoding="onehot",
        ),
        ContinuousParameter(name="a_highpass", min=0.0, max=0.3), # max reduced to avoid total silence

        # Oscillator 1 parameters
        # CategoricalParameter(
        #     name="a_osc_1_type",
        #     values=["Modern"],
        #     raw_values=[0.7083], # Derived from probing (Indices 0 and 8)
        #     encoding="onehot"
        # ), 

        ContinuousParameter(
            name="a_osc_1_pitch", min=0.49, max=0.51, constant_val_p=0.5, constant_val=0.5
        ),
        
        CategoricalParameter(
            name="a_osc_1_route",
            values=["Filter 1"],
            raw_values=[0.1265],
            encoding="onehot",
        ),
        
        ContinuousParameter(name="a_osc_1_shape", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_width_1", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_width_2", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_osc_1_unison_detune", 
            min=0.0, 
            max=0.3, 
            distribution="log"
        ),
        CategoricalParameter(
            name="a_osc_1_unison_voices",
            values=["3 voices"],
            raw_values=[0.137],
            encoding="onehot",
        ),
        ContinuousParameter(name="a_osc_1_volume", min=0.8, max=1.0), # min increased (approx -10dB)
        ContinuousParameter(name="a_osc_1_sub_mix", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_sync", min=0.0, max=1.0),
    
        # noise
        ContinuousParameter(
            name="a_noise_volume", min=0.0, max=1.0, constant_val_p=0.67
        ),
    ],

    [
        DiscreteLiteralParameter(
            name="pitch",
            min=36,
            max=72,
        ),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ]
)