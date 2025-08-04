import whisper
import gradio as gr
import os
from datetime import datetime
import json

# הגדרות
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# טען מודל
print(f"🔄 טוען מודל Whisper {MODEL_SIZE}...")
model = whisper.load_model(MODEL_SIZE)
print("✅ המודל מוכן!")

def transcribe_audio(audio_file, options):
    """תמלל קובץ אודיו עם אפשרויות מתקדמות"""
    if not audio_file:
        return "❌ אנא העלה קובץ", "", ""
    
    try:
        # הגדרות תמלול
        task = "translate" if "תרגום לאנגלית" in options else "transcribe"
        
        # תמלל
        print(f"🎙️ מתמלל: {os.path.basename(audio_file)}")
        result = model.transcribe(
            audio_file,
            language="he" if "עברית" in options else None,
            task=task,
            verbose=True
        )
        
        # שמור קבצים
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # טקסט
        txt_path = f"{OUTPUT_DIR}/{base_name}_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # SRT
        srt_path = f"{OUTPUT_DIR}/{base_name}_{timestamp}.srt"
        create_srt(result["segments"], srt_path)
        
        # JSON
        json_path = f"{OUTPUT_DIR}/{base_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return (
            result["text"],
            f"✅ הקבצים נשמרו ב-{OUTPUT_DIR}/",
            create_preview(result["segments"])
        )
        
    except Exception as e:
        return f"❌ שגיאה: {str(e)}", "", ""

def create_srt(segments, output_path):
    """צור קובץ כתוביות SRT"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")

def format_timestamp(seconds):
    """המר שניות לפורמט SRT"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def create_preview(segments):
    """צור תצוגה מקדימה של הפלחים"""
    preview = "🎬 תצוגה מקדימה (5 פלחים ראשונים):\n\n"
    for i, seg in enumerate(segments[:5], 1):
        time = f"{seg['start']:.1f}s"
        preview += f"[{time}] {seg['text']}\n"
    if len(segments) > 5:
        preview += f"\n... ועוד {len(segments)-5} פלחים"
    return preview

# ממשק Gradio
with gr.Blocks(title="תמלול Whisper", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎙️ תמלול עברית עם Whisper
    
    ### 📌 מדריך מהיר:
    1. העלה קובץ אודיו/וידאו
    2. בחר אפשרויות
    3. לחץ תמלל!
    
    💾 כל הקבצים נשמרים אוטומטית בתיקיית `output/`
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="📁 קובץ אודיו/וידאו"
            )
            
            options = gr.CheckboxGroup(
                choices=[
                    "עברית",
                    "תרגום לאנגלית",
                    "הוסף חותמות זמן"
                ],
                value=["עברית"],
                label="⚙️ אפשרויות"
            )
            
            transcribe_btn = gr.Button(
                "🚀 התחל תמלול",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="📝 תמלול",
                lines=10,
                max_lines=20,
                rtl=True
            )
            
            status = gr.Textbox(
                label="📊 סטטוס",
                lines=1
            )
            
            preview = gr.Textbox(
                label="👁️ תצוגה מקדימה",
                lines=5
            )
    
    # אירועים
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, options],
        outputs=[output_text, status, preview]
    )
    
    # הוראות נוספות
    gr.Markdown("""
    ---
    ### 💡 טיפים:
    - **מודל נוכחי:** {model}
    - **גודל מקסימלי:** 25MB ב-Codespaces
    - **פורמטים נתמכים:** MP3, WAV, MP4, M4A ועוד
    
    ### 📁 קבצי פלט:
    - `.txt` - טקסט נקי
    - `.srt` - כתוביות לוידאו  
    - `.json` - מידע מלא כולל זמנים
    """.format(model=MODEL_SIZE))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True
    )
