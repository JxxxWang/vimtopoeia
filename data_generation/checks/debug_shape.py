
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
from pedalboard import VST3Plugin

def check_param():
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    plugin = load_plugin(plugin_path)
    
    param_name = "a_osc_1_shape"
    if param_name in plugin.parameters:
        print(f"SUCCESS: {param_name} found.")
    else:
        print(f"FAILURE: {param_name} NOT found.")
        
    # Check all types
    types = {
        "Classic": 0.0417,
        "Modern": 0.7083,
        "Wavetable": 0.2083
    }
    
    for name, val in types.items():
        print(f"Setting type to {name} ({val})...")
        if "a_osc_1_type" in plugin.parameters:
            plugin.parameters["a_osc_1_type"].raw_value = val
            
        # Check if shape exists
        if param_name in plugin.parameters:
            print(f"SUCCESS: {param_name} found in {name}.")
        else:
            print(f"FAILURE: {param_name} NOT found in {name}.")
            # List what IS available
            # print(plugin.parameters.keys())

if __name__ == "__main__":
    check_param()
