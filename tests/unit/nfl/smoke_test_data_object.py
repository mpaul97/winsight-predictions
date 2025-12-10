"""Smoke test for DataObject to verify PBP features are loaded correctly."""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, PROJECT_ROOT)

from winsight_predictions.nfl.data_object import DataObject

def test_data_object_columns():
    """Test that DataObject loads all necessary column lists."""
    print("=" * 80)
    print("SMOKE TEST: DataObject PBP Feature Columns")
    print("=" * 80)
    
    # Use absolute path to data storage
    data_storage_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "sports-data-storage-copy"))
    
    print(f"\nData storage path: {data_storage_path}")
    print(f"Path exists: {os.path.exists(data_storage_path)}")
    
    if not os.path.exists(data_storage_path):
        print("\n⚠️  WARNING: Data storage path not found!")
        print(f"   Looking for: {data_storage_path}")
        print("   Trying alternative path...")
        
        # Try alternative path
        alt_path = "G:/My Drive/sports-data-storage-copy"
        if os.path.exists(alt_path):
            data_storage_path = alt_path
            print(f"   ✓ Found alternative path: {data_storage_path}")
        else:
            print(f"   ✗ Alternative path not found: {alt_path}")
            print("\n   Cannot proceed with smoke test - data files not accessible")
            return
    
    data_obj = DataObject(local_root=data_storage_path)
    
    print(f"\n1. Redzone Columns ({len(data_obj.redzone_columns)} total):")
    if data_obj.redzone_columns:
        print(f"   Sample: {data_obj.redzone_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No redzone columns found!")
    
    print(f"\n2. Team EPA Columns ({len(data_obj.team_epa_columns)} total):")
    if data_obj.team_epa_columns:
        print(f"   Sample: {data_obj.team_epa_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No team EPA columns found!")
    
    print(f"\n3. Play Type Columns ({len(data_obj.play_type_columns)} total):")
    if data_obj.play_type_columns:
        print(f"   Sample: {data_obj.play_type_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No play type columns found!")
    
    print(f"\n4. Yards To Go Columns ({len(data_obj.yards_togo_columns)} total):")
    if data_obj.yards_togo_columns:
        print(f"   Sample: {data_obj.yards_togo_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No yards to go columns found!")
    
    print(f"\n5. Yards Gained Columns ({len(data_obj.yards_gained_columns)} total):")
    if data_obj.yards_gained_columns:
        print(f"   Sample: {data_obj.yards_gained_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No yards gained columns found!")
    
    print(f"\n6. Big Play Position Columns ({len(data_obj.big_play_position_columns)} total):")
    if data_obj.big_play_position_columns:
        print(f"   Sample: {data_obj.big_play_position_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No big play position columns found!")
    
    print(f"\n7. Player EPA Position Columns ({len(data_obj.player_epa_position_columns)} total):")
    if data_obj.player_epa_position_columns:
        print(f"   Sample: {data_obj.player_epa_position_columns[:5]}")
    else:
        print("   ⚠️  EMPTY - No player EPA position columns found!")
    
    # Check game data
    print(f"\n8. Game Data Shape: {data_obj.get_game_data_with_features().shape}")
    game_data = data_obj.get_game_data_with_features()
    
    # Verify actual columns exist in game data
    print("\n" + "=" * 80)
    print("VERIFICATION: Do expected columns exist in game data?")
    print("=" * 80)
    
    missing_redzone = [col for col in data_obj.redzone_columns if col not in game_data.columns]
    missing_epa = [col for col in data_obj.team_epa_columns if col not in game_data.columns]
    missing_play_type = [col for col in data_obj.play_type_columns if col not in game_data.columns]
    
    print(f"\nRedzone columns missing from game_data: {len(missing_redzone)}")
    if missing_redzone:
        print(f"   Sample missing: {missing_redzone[:5]}")
    
    print(f"Team EPA columns missing from game_data: {len(missing_epa)}")
    if missing_epa:
        print(f"   Sample missing: {missing_epa[:5]}")
    
    print(f"Play type columns missing from game_data: {len(missing_play_type)}")
    if missing_play_type:
        print(f"   Sample missing: {missing_play_type[:5]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_expected = (
        len(data_obj.redzone_columns) +
        len(data_obj.team_epa_columns) +
        len(data_obj.play_type_columns) +
        len(data_obj.yards_togo_columns) +
        len(data_obj.yards_gained_columns) +
        len(data_obj.big_play_position_columns) +
        len(data_obj.player_epa_position_columns)
    )
    
    total_missing = len(missing_redzone) + len(missing_epa) + len(missing_play_type)
    
    print(f"Total PBP-related columns expected: {total_expected}")
    print(f"Total columns missing from game_data: {total_missing}")
    
    if total_expected == 0:
        print("\n⚠️  CRITICAL ISSUE: No PBP feature columns loaded!")
        print("   This explains why predict_game() has so many missing features.")
        print("\n   Possible causes:")
        print("   - PBP data files not found")
        print("   - DataObject not calling _merge_pbp_into_games()")
        print("   - Column list population logic not executing")
    elif total_missing > 0:
        print(f"\n⚠️  WARNING: {total_missing} columns defined but not in game_data")
    else:
        print("\n✓ All PBP feature columns successfully loaded!")

if __name__ == '__main__':
    test_data_object_columns()
