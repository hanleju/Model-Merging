"""
Transform UTKFace dataset to FairFace format
Renames images and creates label CSV file
"""

import os
import shutil
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm


# =========================================================
# Label Mappings
# =========================================================

def age_to_group(age: int) -> str:
    """Convert age number to age group."""
    if age <= 2:
        return '0-2'
    elif age <= 9:
        return '3-9'
    elif age <= 19:
        return '10-19'
    elif age <= 29:
        return '20-29'
    elif age <= 39:
        return '30-39'
    elif age <= 49:
        return '40-49'
    elif age <= 59:
        return '50-59'
    elif age <= 69:
        return '60-69'
    else:
        return '70+'


def gender_to_label(gender: int) -> str:
    """Convert gender code to label."""
    return 'Male' if gender == 0 else 'Female'


def race_to_label(race: int) -> str:
    """Convert race code to label."""
    race_map = {
        0: 'White',
        1: 'Black',
        2: 'East Asian',
        3: 'Indian',
        4: 'Others'
    }
    return race_map.get(race, 'Others')


# =========================================================
# Parsing Functions
# =========================================================

def parse_utkface_filename(filename: str) -> dict:
    """
    Parse UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    
    Args:
        filename: UTKFace filename
        
    Returns:
        Dictionary with age, gender, race or None if parsing fails
    """
    try:
        parts = filename.split('_')
        if len(parts) < 3:
            return None
        
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        
        # Validate ranges
        if age < 0 or age > 116:  # UTKFace age range
            return None
        if gender not in [0, 1]:
            return None
        if race not in [0, 1, 2, 3, 4]:
            return None
        
        return {
            'age': age,
            'gender': gender,
            'race': race
        }
    except (ValueError, IndexError):
        return None


# =========================================================
# Transformation Functions
# =========================================================

def transform_utkface_to_fairface(
    input_dir: str,
    output_dir: str,
    split_name: str = "train"
):
    """
    Transform UTKFace dataset to FairFace format.
    
    Args:
        input_dir: Directory containing UTKFace images
        output_dir: Output directory for FairFace format
        split_name: Name of the split (train/val/test)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / "images" / split_name
    labels_dir = output_path / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Transforming UTKFace to FairFace format")
    print("="*60)
    print(f"Input      : {input_dir}")
    print(f"Output     : {output_dir}")
    print(f"Split      : {split_name}")
    print("="*60 + "\n")
    
    # Get all image files
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for ext in valid_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    print(f"📁 Found {len(image_files)} image files")
    
    # Parse and transform
    records = []
    skipped = 0
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images"), start=1):
        # Parse filename
        parsed = parse_utkface_filename(image_file.name)
        
        if parsed is None:
            print(f"⚠️  Skipping invalid filename: {image_file.name}")
            skipped += 1
            continue
        
        # Convert labels to FairFace format
        age_group = age_to_group(parsed['age'])
        gender = gender_to_label(parsed['gender'])
        race = race_to_label(parsed['race'])
        
        # New filename
        new_filename = f"{idx}.jpg"
        new_filepath = images_dir / new_filename
        
        # Copy and rename image
        try:
            shutil.copy2(image_file, new_filepath)
        except Exception as e:
            print(f"⚠️  Error copying {image_file.name}: {e}")
            skipped += 1
            continue
        
        # Add record
        records.append({
            'file': f"{split_name}/{new_filename}",
            'age': age_group,
            'gender': gender,
            'race': race
        })
    
    # Create CSV
    df = pd.DataFrame(records)
    csv_path = labels_dir / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Transformation complete!")
    print(f"   Successfully processed: {len(records)}")
    print(f"   Skipped: {skipped}")
    print(f"   Images saved to: {images_dir}")
    print(f"   Labels saved to: {csv_path}")
    
    # Show label distribution
    print(f"\n📊 Label Distribution:")
    print(f"   Age groups:")
    for age, count in df['age'].value_counts().sort_index().items():
        print(f"      {age}: {count}")
    print(f"   Genders:")
    for gender, count in df['gender'].value_counts().items():
        print(f"      {gender}: {count}")
    print(f"   Races:")
    for race, count in df['race'].value_counts().items():
        print(f"      {race}: {count}")


# =========================================================
# Main Function
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transform UTKFace dataset to FairFace format"
    )
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing UTKFace images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for FairFace format dataset")
    parser.add_argument("--split", type=str, default="train",
                       help="Split name (train/val/test) (default: train)")
    
    args = parser.parse_args()
    
    # Check input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Transform dataset
    transform_utkface_to_fairface(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name=args.split
    )


if __name__ == "__main__":
    main()
