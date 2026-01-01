import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys
import numpy as np

def check_osc_types(plugin_path):
    plugin = load_plugin(plugin_path)
    param_name = "a_osc_1_type"
    
    if param_name not in plugin.parameters:
        print(f"Parameter {param_name} not found.")
        return

    param = plugin.parameters[param_name]
    print(f"Probing {param_name}...")
    
    # There are 12 types according to the user. Let's probe 12 steps.
    # We use the center of the bins to be safe.
    # 1/12 = 0.08333...
    # Centers: 0.5/12, 1.5/12, ...
    
    num_types = 12
    for i in range(num_types):
        val = (i + 0.5) / num_types
        param.raw_value = val
        # Pedalboard might not update string_value immediately or might need a specific call, 
        # but usually accessing the property reads from the plugin.
        print(f"Index {i}: Raw={val:.4f}, String='{param.string_value}'")

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_osc_types(plugin_path)
