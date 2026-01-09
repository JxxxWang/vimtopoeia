
import sys
from pathlib import Path
import os
root_dir = Path(".").resolve()
sys.path.append(str(root_dir))
from data_generation import load_plugin

plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
if not os.path.exists(plugin_path):
    print("Plugin not found")
    sys.exit(1)

try:
    plugin = load_plugin(plugin_path)
    
    print("\n--- Inspecting Filter Type ---")
    keys_to_check = ["a_filter_1_type", "a_filter_1_subtype"] # Check subtype too if it exists
    
    for k in keys_to_check:
        if k in plugin.parameters:
            param = plugin.parameters[k]
            print(f"Scanning {k}...")
            
            seen = {}
            # Scan with high resolution
            steps = 1000
            last_str = None
            
            for i in range(steps + 1):
                val = i / steps
                param.raw_value = val
                s = param.string_value
                if s != last_str:
                    print(f"  {k} = '{s}' at raw_value ~ {val:.5f}")
                    seen[s] = val
                    last_str = s
        else:
             print(f"{k} NOT FOUND")

except Exception as e:
    print(f"Error: {e}")
