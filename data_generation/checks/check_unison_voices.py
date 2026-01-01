import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def check_unison_voices(plugin_path):
    plugin = load_plugin(plugin_path)
    param_name = "a_osc_1_unison_voices"
    
    if param_name not in plugin.parameters:
        print(f"Parameter {param_name} not found.")
        return

    param = plugin.parameters[param_name]
    print(f"Probing {param_name}...")
    
    # Probe a few values to find the mapping for 1, 2, 3, 4 voices
    # Usually these are discrete steps.
    steps = 16 
    for i in range(steps + 1):
        val = i / steps
        param.raw_value = val
        print(f"  Val {val:.4f} -> '{param.string_value}'")

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_unison_voices(plugin_path)
