#!/usr/bin/env python3
"""
סקריפט פשוט לתמלול קובץ בודד
שימוש: python simple_transcribe.py <קובץ_אודיו> [גודל_מודל]
"""

import whisper
import sys
import os
import argparse
from pathlib import Path

def main():
    # הגדר פרמטרים
    parser = argparse.ArgumentParser(description='תמלול קובץ אודיו עם Whisper')
    parser.add_argument('audio_file', help='נתיב לקובץ אודיו')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='גודל המודל (ברירת מחדל: base)')
    parser.add_argument('--language', default='he', help='שפה (ברירת מחדל: he)')
    parser.add_argument('--task', default='transcribe', choices=['transcribe', 'translate'],
                       help='משימה: transcribe או translate')
    parser.add_argument('--output', help='נתיב לקובץ פלט (אופציונלי)')
    
    args = parser.parse_args()
    
    # בדוק שהקובץ קיים
    if not os.path.exists(args.audio_file):
        print(f"❌ הקובץ לא נמצא: {args.audio_file}")
        sys.exit(1)
    
    # טען מודל
    print(f"🔄 טוען מודל {args.model}...")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"❌ שגיאה בטעינת המודל: {e}")
        sys.exit(1)
    
    # תמלל
    print(f"🎙️ מתמלל את {args.audio_file}...")
    print(f"   שפה: {args.language}")
    print(f"   משימה: {args.task}")
    
    try:
        result = model.transcribe(
            args.audio_file,
            language=args.language if args.task == 'transcribe' else None,
            task=args.task,
            verbose=False
        )
    except Exception as e:
        print(f"❌ שגיאה בתמלול: {e}")
        sys.exit(1)
    
    # הצג תוצאות
    print("\n" + "="*50)
    print("📝 תוצאת התמלול:")
    print("="*50)
    print(result["text"])
    print("="*50)
    
    # שמור לקובץ
    if args.output:
        output_file = args.output
    else:
        # צור שם קובץ אוטומטי
        base_name = Path(args.audio_file).stem
        output_file = f"{base_name}_transcription.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"\n✅ התמלול נשמר ל: {output_file}")
    
    # הצע ליצור כתוביות
    create_srt = input("\n🎬 ליצור קובץ כתוביות SRT? (כן/לא): ").lower()
    if create_srt in ['כן', 'yes', 'y', 'כ']:
        srt_file = f"{Path(args.audio_file).stem}_subtitles.srt"
        
        with open(srt_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                start = format_time(segment["start"])
                end = format_time(segment["end"])
                f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
        
        print(f"✅ כתוביות נשמרו ל: {srt_file}")

def format_time(seconds):
    """המר שניות לפורמט SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

if __name__ == "__main__":
    main()
