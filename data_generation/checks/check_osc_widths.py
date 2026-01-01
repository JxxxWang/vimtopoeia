import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def check_osc_widths(plugin_path):
    plugin = load_plugin(plugin_path)
    
    def probe_param(type_name, type_val, param_name):
        print(f"\n--- Probing {param_name} in {type_name} (Type={type_val:.4f}) ---")
        plugin.parameters["a_osc_1_type"].raw_value = type_val
        
        param = plugin.parameters[param_name]
        
        steps = 5
        for i in range(steps + 1):
            val = i / steps
            param.raw_value = val
            print(f"  Val {val:.2f} -> '{param.string_value}'")

    # Classic
    probe_param("Classic", 0.0417, "a_osc_1_shape")
    probe_param("Classic", 0.0417, "a_osc_1_width_1") # Maybe this is the sub-shape?
    probe_param("Classic", 0.0417, "a_osc_1_width_2")

    # Modern
    probe_param("Modern", 0.7083, "a_osc_1_shape")
    probe_param("Modern", 0.7083, "a_osc_1_width_1")
    probe_param("Modern", 0.7083, "a_osc_1_width_2")

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_osc_widths(plugin_path)
