#!/usr/bin/env python3
"""
תמלול מרובה קבצים עם Whisper
שימוש: python batch_transcribe.py *.mp3
"""

import whisper
import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import concurrent.futures

def transcribe_file(file_path, model, output_dir):
    """תמלל קובץ בודד"""
    try:
        print(f"\n🎙️ מתמלל: {os.path.basename(file_path)}")
        
        # תמלל
        result = model.transcribe(file_path, language="he")
        
        # צור שמות קבצים
        base_name = Path(file_path).stem
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        
        # שמור טקסט
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # שמור SRT
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start = format_time(segment["start"])
                end = format_time(segment["end"])
                f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
        
        # שמור JSON
        metadata = {
            "file": file_path,
            "date": datetime.now().isoformat(),
            "duration": result.get("duration", 0),
            "text": result["text"],
            "segments": result["segments"]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True, file_path, None
        
    except Exception as e:
        return False, file_path, str(e)

def format_time(seconds):
    """המר שניות לפורמט SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

def main():
    parser = argparse.ArgumentParser(description='תמלול מרובה קבצים')
    parser.add_argument('files', nargs='+', help='קבצי אודיו לתמלול')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='גודל המודל')
    parser.add_argument('--output', default='batch_output', 
                       help='תיקיית פלט')
    parser.add_argument('--parallel', type=int, default=1,
                       help='מספר תמלולים במקביל (ברירת מחדל: 1)')
    
    args = parser.parse_args()
    
    # אסוף קבצים
    all_files = []
    for pattern in args.files:
        if '*' in pattern or '?' in pattern:
            all_files.extend(glob.glob(pattern))
        else:
            all_files.append(pattern)
    
    # סנן רק קבצים קיימים
    valid_files = [f for f in all_files if os.path.exists(f)]
    
    if not valid_files:
        print("❌ לא נמצאו קבצים לתמלול")
        sys.exit(1)
    
    print(f"📁 נמצאו {len(valid_files)} קבצים לתמלול")
    
    # צור תיקיית פלט
    os.makedirs(args.output, exist_ok=True)
    print(f"📂 תיקיית פלט: {args.output}")
    
    # טען מודל
    print(f"\n🔄 טוען מודל {args.model}...")
    model = whisper.load_model(args.model)
    
    # התחל תמלול
    print(f"\n🚀 מתחיל תמלול של {len(valid_files)} קבצים...")
    
    results = []
    failed = []
    
    # תמלול עם progress bar
    with tqdm(total=len(valid_files), desc="תמלול", unit="קובץ") as pbar:
        if args.parallel > 1:
            # תמלול מקבילי
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = []
                for file_path in valid_files:
                    future = executor.submit(transcribe_file, file_path, model, args.output)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    success, file_path, error = future.result()
                    if success:
                        results.append(file_path)
                    else:
                        failed.append((file_path, error))
                    pbar.update(1)
        else:
            # תמלול סדרתי
            for file_path in valid_files:
                success, file_path, error = transcribe_file(file_path, model, args.output)
                if success:
                    results.append(file_path)
                else:
                    failed.append((file_path, error))
                pbar.update(1)
    
    # סיכום
    print("\n" + "="*50)
    print("📊 סיכום תמלול:")
    print("="*50)
    print(f"✅ הצליחו: {len(results)} קבצים")
    print(f"❌ נכשלו: {len(failed)} קבצים")
    
    if failed:
        print("\n🔴 קבצים שנכשלו:")
        for file_path, error in failed:
            print(f"  - {file_path}: {error}")
    
    # צור קובץ סיכום
    summary_file = os.path.join(args.output, "summary.json")
    summary = {
        "date": datetime.now().isoformat(),
        "model": args.model,
        "total_files": len(valid_files),
        "successful": len(results),
        "failed": len(failed),
        "results": results,
        "errors": [{"file": f, "error": e} for f, e in failed]
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 סיכום נשמר ל: {summary_file}")

if __name__ == "__main__":
    main()
