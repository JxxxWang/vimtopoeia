import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def check_filter_drive(plugin_path):
    plugin = load_plugin(plugin_path)
    # Search for "drive" in parameter names
    print("Searching for 'drive' parameters...")
    for name, param in plugin.parameters.items():
        if "drive" in name.lower() and "filter" in name.lower():
            print(f"Found: {name} -> {param.name}")

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    check_filter_drive(plugin_path)
