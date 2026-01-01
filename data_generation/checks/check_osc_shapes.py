import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def check_osc_shapes(plugin_path):
    plugin = load_plugin(plugin_path)
    
    # Helper to probe a parameter while holding type constant
    def probe_shape(type_name, type_val):
        print(f"\n--- Probing {type_name} (Type={type_val:.4f}) ---")
        plugin.parameters["a_osc_1_type"].raw_value = type_val
        
        # Probe Shape Parameter
        shape_param = plugin.parameters["a_osc_1_shape"]
        print(f"Parameter: {shape_param.name}")
        
        # Sweep through 0.0 to 1.0
        steps = 10
        for i in range(steps + 1):
            val = i / steps
            shape_param.raw_value = val
            print(f"  Shape {val:.2f} -> '{shape_param.string_value}'")

    # 1. Probe Classic (Index 0 -> ~0.0417)
    probe_shape("Classic", 0.0417)

    # 2. Probe Modern (Index 8 -> ~0.7083)
    probe_shape("Modern", 0.7083)

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_osc_shapes(plugin_path)
