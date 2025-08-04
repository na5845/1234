#!/usr/bin/env python3
"""
דוגמה בסיסית לתמלול בעברית
"""

import whisper

def main():
    # דוגמה 1: תמלול בסיסי
    print("🎯 דוגמה 1: תמלול בסיסי")
    print("-" * 40)
    
    # טען מודל
    model = whisper.load_model("base")
    
    # תמלל קובץ
    result = model.transcribe("audio.mp3", language="he")
    print(f"תמלול: {result['text']}")
    
    # דוגמה 2: תמלול עם פלחים וזמנים
    print("\n🎯 דוגמה 2: תמלול עם זמנים")
    print("-" * 40)
    
    for segment in result["segments"][:5]:  # 5 פלחים ראשונים
        start = f"{segment['start']:.1f}s"
        end = f"{segment['end']:.1f}s"
        text = segment['text']
        print(f"[{start} - {end}] {text}")
    
    # דוגמה 3: תמלול עם מידע נוסף
    print("\n🎯 דוגמה 3: מידע נוסף")
    print("-" * 40)
    
    # חלץ מידע
    duration = result.get("duration", 0)
    language = result.get("language", "he")
    segments_count = len(result["segments"])
    
    print(f"משך: {duration:.1f} שניות")
    print(f"שפה: {language}")
    print(f"מספר פלחים: {segments_count}")
    
    # דוגמה 4: חיפוש מילים ספציפיות
    print("\n🎯 דוגמה 4: חיפוש מילים")
    print("-" * 40)
    
    search_word = "שלום"  # שנה למילה שאתה מחפש
    found_segments = []
    
    for segment in result["segments"]:
        if search_word in segment['text']:
            found_segments.append(segment)
    
    print(f"נמצאו {len(found_segments)} פלחים עם המילה '{search_word}':")
    for seg in found_segments[:3]:  # הצג 3 ראשונים
        print(f"  [{seg['start']:.1f}s] {seg['text']}")
    
    # דוגמה 5: יצוא לפורמטים שונים
    print("\n🎯 דוגמה 5: יצוא לפורמטים")
    print("-" * 40)
    
    # VTT format
    with open("output.vtt", "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start = format_time_vtt(segment["start"])
            end = format_time_vtt(segment["end"])
            f.write(f"{start} --> {end}\n{segment['text']}\n\n")
    print("✅ נוצר קובץ output.vtt")
    
    # JSON format
    import json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ נוצר קובץ output.json")

def format_time_vtt(seconds):
    """פורמט זמן ל-VTT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

if __name__ == "__main__":
    main()
