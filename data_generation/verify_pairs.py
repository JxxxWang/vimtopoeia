import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
import os
import numpy as np
from pedalboard.io import AudioFile
from data_generation import load_plugin, render_params
from data_generation.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC

def verify_pairs():
    print("Initializing verification...")
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    plugin = load_plugin(plugin_path)
    
    os.makedirs("debug_output", exist_ok=True)
    
    # Generate 3 pairs
    for i in range(3):
        print(f"\n--- Pair {i} ---")
        target_params, ref_params, note_params = SURGE_SIMPLE_PARAM_SPEC.sample_pair()
        
        # Check Categorical Parameters
        print("Categorical Parameters Check:")
        
        # Unison Voices
        t_unison = target_params.get("a_osc_1_unison_voices", "N/A")
        r_unison = ref_params.get("a_osc_1_unison_voices", "N/A")
        print(f"  Unison Voices - Target: {t_unison}, Reference: {r_unison}")
        if t_unison != "N/A" and r_unison != "N/A":
             if abs(t_unison - r_unison) < 0.0001:
                 print("    STATUS: Unison Voices are LOCKED (Correct)")
             else:
                 print("    STATUS: Unison Voices are DIFFERENT (Incorrect if locking is expected)")

        # Osc Type
        t_type = target_params.get("a_osc_1_type", "N/A")
        r_type = ref_params.get("a_osc_1_type", "N/A")
        print(f"  Osc Type      - Target: {t_type}, Reference: {r_type}")
        if t_type != "N/A" and r_type != "N/A":
             if abs(t_type - r_type) < 0.0001:
                 print("    STATUS: Osc Type is LOCKED (Correct)")
             else:
                 print("    STATUS: Osc Type is DIFFERENT (Incorrect if locking is expected)")

        
        # Render Audio
        print("Rendering Target...")
        target_audio = render_params(
            plugin, target_params, note_params["pitch"], 100, 
            note_params["note_start_and_end"], 4.0, 44100, 2
        )
        
        print("Rendering Reference...")
        ref_audio = render_params(
            plugin, ref_params, note_params["pitch"], 100, 
            note_params["note_start_and_end"], 4.0, 44100, 2
        )
        
        # Save Audio
        with AudioFile(f"debug_output/pair_{i}_target.wav", "w", 44100, 2) as f:
            f.write(target_audio.T)
        with AudioFile(f"debug_output/pair_{i}_ref.wav", "w", 44100, 2) as f:
            f.write(ref_audio.T)
            
        # Save Params
        param_dump = {
            "target": target_params,
            "reference": ref_params,
            "note": note_params
        }
        
        # Convert numpy types to float for JSON serialization
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
            
        with open(f"debug_output/pair_{i}_params.json", "w") as f:
            json.dump(param_dump, f, indent=2, default=convert)
            
        print(f"Saved pair_{i} to debug_output/")

if __name__ == "__main__":
    verify_pairs()
