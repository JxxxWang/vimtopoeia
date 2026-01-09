import numpy as np
import rootutils
from loguru import logger
import soundfile as sf
import os
import copy
import json

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin, render_params
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC

def main():
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    output_dir = "checks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Plugin
    print(f"Loading plugin from {plugin_path}...")
    plugin = load_plugin(plugin_path)
    
    # 1. Sample base parameters from the spec
    # We just need one set of valid parameter values to serve as the background
    target, _, note_params = SURGE_SIMPLE_PARAM_SPEC.sample_pair()
    
    # Define common render settings
    render_kwargs = {
        "midi_note": 60, # MIDI note 60 (C4)
        "velocity": 100,
        "note_start_and_end": (0.1, 3.9),
        "signal_duration_seconds": 4.0,
        "sample_rate": 44100.0,
        "channels": 2, # Stereo required by Surge XT
    }
    
    # 2. Create Pure Saw parameter set
    saw_params = copy.deepcopy(target)
    square_params = copy.deepcopy(target)
    
    # Validating Spec vs Plugin Reality:
    # The Spec provides 'a_osc_1_width_1' but the Plugin (Modern) uses 'a_osc_1_width'.
    # references: 'a_osc_1_width' in saw_params is critical.

    
    for p in [saw_params, square_params]:
        # AUDIBILITY ENFORCEMENT
        p["a_volume"] = 1.0             # Master Volume Max (0dB)
        p["a_osc_1_volume"] = 1.0       # Osc 1 Volume Max
        p["a_filter_1_cutoff"] = 1.0    # Filter Fully Open
        p["a_amp_eg_sustain"] = 1.0     # Full Sustain
        p["a_amp_eg_decay"] = 0.5       # Moderate Decay
        p["a_amp_eg_release"] = 0.5     # Moderate Release
        p["a_osc_1_mute"] = 0.0         # Ensure Unmuted
        p["a_osc_2_volume"] = 0.0 
        p["a_osc_3_volume"] = 0.0 

        # p["a_osc_1_type"] = 0.7083  # Modern Oscillator Mode        

    # Modern Oscillator Mode Logic
    # In Modern mode:
    # Shape 0.0 = Sawtooth
    # Shape 0.5 = Square (Pulse)
    
    # Ensure Modern mode is active (if not already)
    # The 'value' from spec suggests Modern is active. 
    # We won't force 'a_osc_1_type' here assuming spec is correct, 
    # but we will set the shape.

    # Pure Saw
    saw_params["a_osc_1_shape"] = 1
    # Reset other morph/timbre params to neutral
    saw_params["a_osc_1_width_1"] = 0.5
    
    # 3. Create Pure Square parameter set
    # square_params initialized above
    
    # Pure Square
    square_params["a_osc_1_shape"] = 0.5
    square_params["a_osc_1_width_1"] = 1 # 50% Duty Cycle

    # Render Saw
    print("Rendering Sawtooth...")
    # Update plugin params logic handles "not found" gracefully
    saw_audio = render_params(plugin, saw_params, **render_kwargs)
    sf.write(os.path.join(output_dir, "pure_saw.wav"), saw_audio.T, 44100)
    
    # Dump Saw Parameters
    saw_full_state = {k: p.raw_value for k, p in plugin.parameters.items()}
    with open(os.path.join(output_dir, "pure_saw_params.json"), "w") as f:
        json.dump(
            {"spec": saw_params, "full": saw_full_state}, 
            f, 
            indent=2, 
            default=lambda o: o.item() if isinstance(o, np.generic) else o
        )
    
    # Render Square
    print("Rendering Square...")
    square_audio = render_params(plugin, square_params, **render_kwargs)
    sf.write(os.path.join(output_dir, "pure_square.wav"), square_audio.T, 44100)
    
    # Dump Square Parameters
    square_full_state = {k: p.raw_value for k, p in plugin.parameters.items()}
    with open(os.path.join(output_dir, "pure_square_params.json"), "w") as f:
        json.dump(
            {"spec": square_params, "full": square_full_state}, 
            f, 
            indent=2, 
            default=lambda o: o.item() if isinstance(o, np.generic) else o
        )
    
    print("Done!")

if __name__ == "__main__":
    main()
