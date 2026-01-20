"""
Test script to verify VimSketch vocal/synth pairing logic.
"""
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from dataset_dual import DualDataset


def test_sketch_pairs(vimsketch_root: str | Path):
    """
    Test the _build_sketch_pairs method to verify correct pairing.
    
    Args:
        vimsketch_root: Path to vimsketch_synth folder
    """
    vimsketch_root = Path(vimsketch_root)
    vocal_dir = vimsketch_root / 'vocal'
    synth_dir = vimsketch_root / 'synth'
    
    print("=" * 80)
    print("VimSketch Vocal/Synth Pairing Test")
    print("=" * 80)
    
    # Check directories exist
    if not vocal_dir.exists():
        print(f"❌ Vocal directory not found: {vocal_dir}")
        return
    if not synth_dir.exists():
        print(f"❌ Synth directory not found: {synth_dir}")
        return
    
    print(f"✓ Vocal directory: {vocal_dir}")
    print(f"✓ Synth directory: {synth_dir}")
    print()
    
    # Get file counts
    vocal_files = list(vocal_dir.glob('*.wav'))
    synth_files = list(synth_dir.glob('*.wav'))
    
    print(f"Found {len(vocal_files)} vocal files")
    print(f"Found {len(synth_files)} synth files")
    print()
    
    # Build pairs using DualDataset's method
    # Create minimal instance that only initializes vimsketch_root and calls _build_sketch_pairs
    try:
        class MinimalPairTester(DualDataset):
            """Minimal version that only tests pairing without loading H5 file"""
            def __init__(self, vimsketch_root):
                self.vimsketch_root = Path(vimsketch_root)
                # Call only the pairing method, skip everything else
                self.pairs = self._build_sketch_pairs()
        
        tester = MinimalPairTester(vimsketch_root)
        pairs = tester.pairs
        
        print(f"✓ Successfully built {len(pairs)} pairs")
        print()
        
        # Show sample pairs
        print("=" * 80)
        print("Sample Pairs (first 10):")
        print("=" * 80)
        for i, (vocal_path, synth_path) in enumerate(pairs[:10]):
            print(f"\nPair {i}:")
            print(f"  Vocal: {vocal_path.name}")
            print(f"  Synth: {synth_path.name}")
        
        if len(pairs) > 10:
            print(f"\n... ({len(pairs) - 10} more pairs)")
        
        # Statistics
        print()
        print("=" * 80)
        print("Pairing Statistics:")
        print("=" * 80)
        
        # Count how many synths are used
        unique_synths = set(synth_path for _, synth_path in pairs)
        print(f"Total pairs: {len(pairs)}")
        print(f"Unique synths used: {len(unique_synths)}")
        print(f"Average vocals per synth: {len(pairs) / len(unique_synths):.2f}")
        
        # Show synth usage distribution
        synth_usage = {}
        for _, synth_path in pairs:
            synth_name = synth_path.name
            synth_usage[synth_name] = synth_usage.get(synth_name, 0) + 1
        
        print()
        print("Top 5 most used synths:")
        sorted_usage = sorted(synth_usage.items(), key=lambda x: x[1], reverse=True)
        for synth_name, count in sorted_usage[:5]:
            print(f"  {synth_name}: {count} vocals")
        
        # Verify naming conventions
        print()
        print("=" * 80)
        print("Naming Convention Verification:")
        print("=" * 80)
        
        vocal_format_ok = 0
        synth_format_ok = 0
        
        for vocal_path, synth_path in pairs:
            vocal_name = vocal_path.stem
            synth_name = synth_path.stem
            
            # Check vocal format: 5 digits + underscore
            vocal_parts = vocal_name.split('_', 1)
            if len(vocal_parts) == 2 and len(vocal_parts[0]) == 5 and vocal_parts[0].isdigit():
                vocal_format_ok += 1
            
            # Check synth format: 3 digits + underscore
            synth_parts = synth_name.split('_', 1)
            if len(synth_parts) == 2 and len(synth_parts[0]) == 3 and synth_parts[0].isdigit():
                synth_format_ok += 1
        
        print(f"✓ Vocals with correct format (5-digit prefix): {vocal_format_ok}/{len(pairs)}")
        print(f"✓ Synths with correct format (3-digit prefix): {synth_format_ok}/{len(pairs)}")
        
    except Exception as e:
        print(f"❌ Error building pairs: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Default path
    vimsketch_root = Path(__file__).parent.parent.parent / 'vimsketch_synth'
    
    if not vimsketch_root.exists():
        print(f"VimSketch root not found at: {vimsketch_root}")
        print("Please provide the correct path as an argument:")
        print(f"  python {Path(__file__).name} /path/to/vimsketch_synth")
        sys.exit(1)
    
    test_sketch_pairs(vimsketch_root)
