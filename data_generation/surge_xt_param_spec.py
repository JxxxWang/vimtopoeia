from data_generation.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

SURGE_SIMPLE_PARAM_SPEC = ParamSpec(
    [
        ContinuousParameter(name="a_amp_eg_attack", min=0.0, max=0.15, distribution="log"),  
        ContinuousParameter(name="a_amp_eg_decay", min=0.0, max=0.77, distribution="log"),  # max around 4s
        ContinuousParameter(
            name="a_amp_eg_release", min=0.0, max=0.77, distribution="log"
        ),  # max around 4s
        ContinuousParameter(name="a_amp_eg_sustain", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_1_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_1_feg_mod_amount", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_1_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_2_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_2_feg_mod_amount", min=0.0, max=1.0),
        ContinuousParameter(name="a_filter_2_resonance", min=0.0, max=1.0),
        ContinuousParameter(
            name="a_filter_eg_attack", min=0.0, max=0.77
        ),  # max around 4s
        ContinuousParameter(
            name="a_filter_eg_decay", min=0.0, max=0.77
        ),  # max around 4s
        ContinuousParameter(
            name="a_filter_eg_release", min=0.0, max=0.77
        ),  # max around 4s
        ContinuousParameter(name="a_filter_eg_sustain", min=0.0, max=1.0),
        ContinuousParameter(name="a_highpass", min=0.0, max=1.0, constant_val_p=0.5),
        ContinuousParameter(
            name="a_noise_volume", min=0.0, max=1.0, constant_val_p=0.67
        ),

        CategoricalParameter(
            name="a_osc_1_type",
            values=["Modern"],
            raw_values=[0.7083], # Derived from probing (Indices 0 and 8)
            encoding="onehot"),

        ContinuousParameter(
            name="a_osc_1_pitch", min=0.0, max=1.0, constant_val_p=0.5, constant_val=0.5
        ),
        ContinuousParameter(name="a_osc_1_sawtooth", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_pulse", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_triangle", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_width", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_sync", min=0.0, max=1.0),
        ContinuousParameter(name="a_osc_1_volume", min=0.5, max=1.0),
        
        CategoricalParameter(
            name="a_osc_1_unison_voices",
            values=[
                "1 voice",
                "2 voices",
                "3 voices",
                "4 voices",
            ],
            raw_values=[
                0.0195,
                0.07150000000000001,
                0.137,
                0.203,
            ],
            weights=[3.0, 1, 1, 0],
            encoding="onehot",
        ),
        ContinuousParameter(name="a_osc_1_unison_detune", min=0.0, max=0.5),
        
        ContinuousParameter(
            name="a_lfo_1_amplitude", min=0.0, max=1.0, constant_val_p=0.67
        ),
        ContinuousParameter(name="a_lfo_1_attack", min=0.0, max=0.77),
        ContinuousParameter(name="a_lfo_1_decay", min=0.0, max=0.77),
        ContinuousParameter(name="a_lfo_1_deform", min=0.0, max=1.0),
        ContinuousParameter(name="a_lfo_1_hold", min=0.0, max=0.77),
        ContinuousParameter(name="a_lfo_1_phase", min=0.0, max=1.0),
        ContinuousParameter(name="a_lfo_1_rate", min=0.0, max=1.0),
        ContinuousParameter(name="a_lfo_1_release", min=0.0, max=0.77),
        ContinuousParameter(name="a_lfo_1_sustain", min=0.0, max=1.0),
        ],
    [
        DiscreteLiteralParameter(
            name="pitch",
            min=36,
            max=72,
        ),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ],
)

# === VIMTOPOEIA V2.0: The "Slim" Spec ===
# Focus: Reference-Guided Differential Learning
# Strategy: Only learn the parameters that define the "Voice" (Timbre, Articulation, Dynamics).
# Everything else is locked down to ensure a stable topology.


# not using this spec currently
# SURGE_XT_PARAM_SPEC = ParamSpec(
#     # --- 1. ACTIVE PARAMETERS (The Model Predicts These) ---
#     params=[
#         # --- Oscillator 1 (The "Voice") ---
#         # Type: Classic (Analog) vs Modern (Clean/Digital)
#         # We include both to capture different "flavors" of basic waveforms.
#         # Note: This is locked between Target/Reference by the sampling logic.
#         # CategoricalParameter(
#         #     name="a_osc_1_type",
#         #     values=["Classic", "Modern", "Wavetable"],
#         #     raw_values=[0.0417, 0.7083, 0.2083], # Derived from probing (Indices 0 and 8)
#         #     encoding="onehot",
#         # ),

#         CategoricalParameter(
#             name="a_osc_1_unison_voices",
#             values=[
#                 "1 voice",
#                 "2 voices",
#                 "3 voices",
#                 "4 voices",
#             ],
#             raw_values=[
#                 0.0195,
#                 0.07150000000000001,
#                 0.137,
#                 0.203,
#             ],
#             weights=[3.0, 1, 1, 0],
#             encoding="onehot",
#         ),
        
#         # Shape: The core timbre (Saw -> Square -> Pulse etc.) OR Wavetable Position
#         ContinuousParameter(name="a_osc_1_shape", min=0.0, max=1.0),
#         # Width 1: Pulse width (Classic) or Formant/Spectrum (Wavetable)
#         ContinuousParameter(name="a_osc_1_width_1", min=0.0, max=1.0),
#         # Width 2: Sub-oscillator (Classic) or LP Filter/Spectrum (Wavetable)
#         ContinuousParameter(name="a_osc_1_width_2", min=0.0, max=1.0),
        
#         # Unison: Thickness / Chorus effect
#         ContinuousParameter(name="a_osc_1_unison_detune", min=0.0, max=0.5),
        
#         # Feedback: Adds grit/texture (Global feedback loop)
#         ContinuousParameter(name="a_feedback", min=0.0, max=0.6),

#         # --- Filter 1 (The "Mouth") ---
#         # Cutoff: The main vowel/ control
#         ContinuousParameter(name="a_filter_1_cutoff", min=0.1, max=0.9),
#         # Resonance: Nasality / Formant peak
#         ContinuousParameter(name="a_filter_1_resonance", min=0.0, max=0.75),
#         # Drive: Saturation/Grit (Note: Surge XT might not have a direct 'drive' param for all filter types, 
#         # but 'waveshaper_drive' is a global distortion often used with filters)
#         ContinuousParameter(name="a_waveshaper_drive", min=0.0, max=0.6),

#         # --- Amp Envelope (The "Lungs" / Dynamics) ---
#         # How the sound starts and stops
#         # Use Log-Uniform distribution for time parameters as requested
#         ContinuousParameter(name="a_amp_eg_attack", min=0.0, max=0.77, distribution="log"),
#         ContinuousParameter(name="a_amp_eg_decay", min=0.001, max=0.77, distribution="log"),
#         ContinuousParameter(name="a_amp_eg_sustain", min=0.0, max=1.0),
#         ContinuousParameter(name="a_amp_eg_release", min=0.001, max=0.77, distribution="log"),

#         # --- Filter Envelope (The "Articulation") ---
#         # How the "Mouth" moves over time (e.g., "Wah", "Yoi")
#         ContinuousParameter(name="a_filter_eg_attack", min=0.0, max=0.77, distribution="log"),
#         ContinuousParameter(name="a_filter_eg_decay", min=0.0, max=0.77, distribution="log"),
#         ContinuousParameter(name="a_filter_eg_sustain", min=0.0, max=1.0),
#         ContinuousParameter(name="a_filter_eg_release", min=0.001, max=0.77, distribution="log"),
#         # Modulation Amount: How much the envelope moves the filter cutoff
#         ContinuousParameter(name="a_filter_1_feg_mod_amount", min=0.0, max=1.0),

#         # --- Mixer (Balance) ---
#         # Oscillator 1 Level
#         ContinuousParameter(name="a_osc_1_volume", min=0.5, max=1.0),
#         # Noise Level: Breathiness / Air
#         ContinuousParameter(name="a_noise_volume", min=0.0, max=0.5),
#     ],
    
#     # --- 2. STATIC PARAMETERS (Fixed Topology) ---
#     # These are enforced on EVERY patch to ensure consistency.
#     # We disable unused oscillators and effects to keep the signal path clean.
#     fixed_params={
#         # --- Oscillator 1 Safety Defaults ---
#         "a_osc_1_type": 0.7083, # Fixed to Modern
#         "a_osc_1_mute": 0.0,   # Ensure not muted
#         "a_osc_1_solo": 0.0,
#         "a_osc_1_octave": 0.5, # Center (0 octave shift)
#         "a_osc_1_pitch": 0.5,  # Center (0 semitone shift)
        
#         # Unison Setup: Fixed to 3 voices.
#         # This allows 'a_osc_1_unison_detune' to act as a continuous "Thickness" control.
#         # If detune is 0, it sounds like 1 voice. If detune > 0, it sounds like a chorus.
#         # "a_osc_1_unison_voices": 0.125, # Moved to Active Params
        
#         # --- Filter 1 Setup ---
#         # Force a standard Lowpass filter so Cutoff/Resonance always work
#         "a_filter_1_type": 0.0,    # Lowpass
#         "a_filter_1_subtype": 0.0, # Standard/Analog
        
#         # --- Disable Extra Oscillators ---
#         "a_osc_2_volume": 0.0,
#         "a_osc_3_volume": 0.0,
#         "a_osc_2_mute": 1.0,
#         "a_osc_3_mute": 1.0,
        
#         # --- Noise Enabled (Controlled by Active Params) ---
#         "a_noise_mute": 0.0,
#         "a_noise_color": 0.5, # Fixed to neutral
        
#         # --- Disable Ring Mod ---
#         "a_ring_modulation_1x2_volume": 0.0,
#         "a_ring_modulation_2x3_volume": 0.0,
#         "a_ring_modulation_1x2_mute": 1.0,
#         "a_ring_modulation_2x3_mute": 1.0,

#         # --- Disable Scene B ---
#         "b_volume": 0.0,
#         # "b_mute": 1.0, # Parameter does not exist
        
#         # --- Global Settings ---
#         "global_volume": 1.0,
#         "a_filter_configuration": 0.0, # Serial 1
        
#     },

#     # --- 3. NOTE PARAMETERS ---
#     note_params=[
#         # Range: C2 (36) to C5 (84).
#         DiscreteLiteralParameter(name="pitch", min=36, max=72),
#         NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
#     ]
# )
