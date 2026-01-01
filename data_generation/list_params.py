import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation import load_plugin
import sys

def list_params(plugin_path):
    plugin = load_plugin(plugin_path)
    for name, param in plugin.parameters.items():
        print(f"'{name}': {param.raw_value}")

if __name__ == "__main__":
    plugin_path = "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"
    if len(sys.argv) > 1:
        plugin_path = sys.argv[1]
    list_params(plugin_path)
