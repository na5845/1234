#!/usr/bin/env python3
"""
דוגמאות מתקדמות לשימוש ב-Whisper
"""

import whisper
import numpy as np
import torch
from typing import List, Dict, Tuple
import re
from collections import Counter

class AdvancedTranscriber:
    """מחלקה מתקדמת לתמלול עם תכונות נוספות"""
    
    def __init__(self, model_size="base"):
        print(f"🔄 טוען מודל {model_size}...")
        self.model = whisper.load_model(model_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ משתמש ב: {self.device}")
    
    def transcribe_with_confidence(self, audio_file: str) -> Dict:
        """תמלול עם רמת ביטחון לכל פלח"""
        result = self.model.transcribe(
            audio_file, 
            language="he",
            temperature=0,  # יותר דטרמיניסטי
            best_of=5,      # נסה 5 פעמים
            beam_size=5     # חיפוש קרן רחב יותר
        )
        
        # הוסף ניתוח לכל פלח
        for segment in result["segments"]:
            # חשב רמת ביטחון (פשוט לדוגמה)
            segment["confidence"] = self._calculate_confidence(segment)
            
            # מצא מילים ארוכות
            segment["long_words"] = self._find_long_words(segment["text"])
        
        return result
    
    def _calculate_confidence(self, segment: Dict) -> float:
        """חשב רמת ביטחון לפלח (דמה)"""
        # בפועל, Whisper לא מחזיר confidence scores
        # זה רק דוגמה לאיך אפשר להוסיף ניתוח
        text_length = len(segment["text"])
        duration = segment["end"] - segment["start"]
        
        # הנחה: טקסט ארוך יותר בזמן קצר = פחות אמין
        if duration > 0:
            chars_per_second = text_length / duration
            confidence = min(1.0, 10.0 / chars_per_second)
        else:
            confidence = 0.5
        
        return round(confidence, 2)
    
    def _find_long_words(self, text: str, min_length: int = 6) -> List[str]:
        """מצא מילים ארוכות בטקסט"""
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) >= min_length]
    
    def create_summary(self, result: Dict) -> Dict:
        """צור סיכום מתקדם של התמלול"""
        text = result["text"]
        segments = result["segments"]
        
        # ניתוח בסיסי
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        
        # מילים נפוצות
        word_freq = Counter(words)
        common_words = word_freq.most_common(10)
        
        # פלחים ארוכים
        long_segments = sorted(
            segments, 
            key=lambda s: s["end"] - s["start"], 
            reverse=True
        )[:5]
        
        # פלחים עם ביטחון נמוך (אם יש)
        low_confidence = [
            s for s in segments 
            if s.get("confidence", 1.0) < 0.5
        ]
        
        summary = {
            "total_duration": segments[-1]["end"] if segments else 0,
            "total_words": len(words),
            "total_sentences": len([s for s in sentences if s.strip()]),
            "average_words_per_minute": len(words) / (segments[-1]["end"] / 60) if segments else 0,
            "most_common_words": common_words,
            "longest_segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "duration": s["end"] - s["start"],
                    "text": s["text"][:50] + "..."
                }
                for s in long_segments
            ],
            "low_confidence_count": len(low_confidence)
        }
        
        return summary
    
    def extract_speakers(self, result: Dict) -> List[Dict]:
        """נסיון לזהות דוברים שונים (בסיסי)"""
        # Whisper לא מזהה דוברים, אבל אפשר לנסות לזהות
        # לפי הפסקות ארוכות או שינויים בסגנון
        
        segments = result["segments"]
        speakers = []
        current_speaker = 1
        last_end = 0
        
        for segment in segments:
            # אם יש הפסקה ארוכה, אולי זה דובר חדש
            gap = segment["start"] - last_end
            if gap > 3.0:  # הפסקה של 3 שניות
                current_speaker += 1
            
            speakers.append({
                "speaker": f"דובר {current_speaker}",
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
            
            last_end = segment["end"]
        
        return speakers
    
    def export_for_editing(self, result: Dict, output_file: str):
        """יצוא בפורמט מתאים לעריכת וידאו"""
        with open(output_file, "w", encoding="utf-8") as f:
            # כותרת
            f.write("# תמלול לעריכה\n")
            f.write(f"# נוצר עם Whisper\n\n")
            
            # מטא-דאטה
            duration = result["segments"][-1]["end"] if result["segments"] else 0
            f.write(f"משך כולל: {self._format_duration(duration)}\n")
            f.write(f"מספר פלחים: {len(result['segments'])}\n\n")
            
            # פלחים עם טיימקודים
            f.write("## טיימליין\n\n")
            for i, segment in enumerate(result["segments"], 1):
                start = self._format_duration(segment["start"])
                end = self._format_duration(segment["end"])
                
                f.write(f"### [{i:03d}] {start} - {end}\n")
                f.write(f"{segment['text']}\n\n")
                
                # הוסף הערות אם יש
                if segment.get("confidence", 1.0) < 0.5:
                    f.write(f"⚠️ **הערה:** רמת ביטחון נמוכה\n\n")
    
    def _format_duration(self, seconds: float) -> str:
        """פורמט זמן לקריאה נוחה"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

def main():
    """דוגמאות לשימוש מתקדם"""
    
    # צור מתמלל מתקדם
    transcriber = AdvancedTranscriber("base")
    
    # דוגמה 1: תמלול עם ניתוח
    print("🎯 תמלול עם ניתוח מתקדם")
    print("-" * 50)
    
    # החלף עם הקובץ שלך
    audio_file = "example.mp3"
    
    try:
        # תמלל עם תכונות נוספות
        result = transcriber.transcribe_with_confidence(audio_file)
        
        # צור סיכום
        summary = transcriber.create_summary(result)
        
        print(f"📊 סיכום התמלול:")
        print(f"  • משך: {summary['total_duration']:.1f} שניות")
        print(f"  • מילים: {summary['total_words']}")
        print(f"  • משפטים: {summary['total_sentences']}")
        print(f"  • מילים לדקה: {summary['average_words_per_minute']:.1f}")
        
        print(f"\n📈 מילים נפוצות:")
        for word, count in summary['most_common_words'][:5]:
            print(f"  • {word}: {count} פעמים")
        
        # דוגמה 2: זיהוי דוברים
        print("\n🎯 ניסיון לזיהוי דוברים")
        print("-" * 50)
        
        speakers = transcriber.extract_speakers(result)
        speaker_counts = Counter(s["speaker"] for s in speakers)
        
        print(f"נמצאו {len(speaker_counts)} דוברים אפשריים:")
        for speaker, count in speaker_counts.items():
            print(f"  • {speaker}: {count} פלחים")
        
        # דוגמה 3: יצוא לעריכה
        print("\n🎯 יצוא לעריכת וידאו")
        print("-" * 50)
        
        output_file = "transcript_for_editing.md"
        transcriber.export_for_editing(result, output_file)
        print(f"✅ נוצר קובץ: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ הקובץ {audio_file} לא נמצא")
        print("💡 שנה את שם הקובץ בקוד לקובץ קיים")

if __name__ == "__main__":
    main()
