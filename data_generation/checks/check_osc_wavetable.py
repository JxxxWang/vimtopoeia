import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def check_osc_wavetable(plugin_path):
    plugin = load_plugin(plugin_path)
    
    # Set to Wavetable type
    wavetable_type_val = 0.2083
    print(f"\n--- Setting Oscillator Type to Wavetable ({wavetable_type_val}) ---")
    plugin.parameters["a_osc_1_type"].raw_value = wavetable_type_val
    
    # Helper to probe a parameter
    def probe_param(param_name):
        if param_name not in plugin.parameters:
            print(f"Parameter {param_name} not found.")
            return

        param = plugin.parameters[param_name]
        print(f"\nProbing {param_name} ({param.name}):")
        
        steps = 5
        for i in range(steps + 1):
            val = i / steps
            param.raw_value = val
            print(f"  Val {val:.2f} -> '{param.string_value}'")

    # Probe standard morph parameters
    probe_param("a_osc_1_shape")
    probe_param("a_osc_1_width_1")
    probe_param("a_osc_1_width_2") # Often sub-osc or unison related in other modes
    
    # Probe potential wavetable selector. 
    # In some synths, "Shape" scans the table, and another param selects the table.
    # Or "Shape" selects the table? Let's see the string values.

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_osc_wavetable(plugin_path)
