# app.py - אפליקציית תמלול עם ממשק ויזואלי
import whisper
import gradio as gr
import os
from datetime import datetime

# בחר גודל מודל לפי המשאבים
# tiny = 1GB, base = 1.5GB, small = 2.5GB
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
print(f"טוען מודל Whisper {MODEL_SIZE}...")
model = whisper.load_model(MODEL_SIZE)
print("המודל נטען בהצלחה!")

def transcribe_audio(audio_file, task_type):
    """תמלל קובץ אודיו"""
    if audio_file is None:
        return "❌ אנא העלה קובץ אודיו", ""
    
    try:
        print(f"מתחיל תמלול של: {audio_file}")
        
        # תמלל
        result = model.transcribe(
            audio_file, 
            language="he" if task_type == "תמלול בעברית" else None,
            task="transcribe" if task_type != "תרגום לאנגלית" else "translate"
        )
        
        # טקסט מלא
        full_text = result["text"].strip()
        
        # צור כתוביות SRT
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_time_srt(segment["start"])
            end_time = format_time_srt(segment["end"])
            srt_content += f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"
        
        # שמור לקבצים
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # שמור טקסט
        txt_filename = f"transcription_{timestamp}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        # שמור SRT
        srt_filename = f"subtitles_{timestamp}.srt"
        with open(srt_filename, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        download_links = f"\n\n📥 קבצים להורדה:\n"
        download_links += f"- [הורד טקסט](./{txt_filename})\n"
        download_links += f"- [הורד כתוביות SRT](./{srt_filename})"
        
        return full_text + download_links, srt_content
        
    except Exception as e:
        return f"❌ שגיאה: {str(e)}", ""

def format_time_srt(seconds):
    """המר שניות לפורמט SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

# צור ממשק Gradio
with gr.Blocks(title="תמלול Whisper", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎙️ תמלול עם Whisper ב-GitHub Codespaces
    
    ### 📌 הוראות:
    1. העלה קובץ אודיו או וידאו (עד 25MB)
    2. בחר שפה/משימה
    3. לחץ על "תמלל" והמתן
    
    💡 **טיפ:** תמלול של 10 דקות לוקח בערך 2-3 דקות
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                source="upload", 
                type="filepath",
                label="📁 העלה קובץ אודיו/וידאו"
            )
            task_type = gr.Radio(
                choices=["תמלול בעברית", "תמלול אוטומטי (כל שפה)", "תרגום לאנגלית"],
                value="תמלול בעברית",
                label="🌐 בחר משימה"
            )
            transcribe_btn = gr.Button("🚀 תמלל", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="📝 תוצאת התמלול",
                lines=10,
                rtl=True,
                interactive=True
            )
            output_srt = gr.Textbox(
                label="🎬 כתוביות SRT",
                lines=5,
                visible=False
            )
    
    # חיבור הפונקציה
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, task_type],
        outputs=[output_text, output_srt]
    )
    
    gr.Markdown("""
    ---
    ### 🛠️ מידע טכני:
    - **מודל:** Whisper {model_size}
    - **סביבה:** GitHub Codespaces
    - **זיכרון פנוי:** {memory_info}
    """.format(
        model_size=MODEL_SIZE,
        memory_info="בדיקה..."
    ))

if __name__ == "__main__":
    # הגדרות להרצה ב-Codespaces
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 מפעיל את האפליקציה על פורט {port}")
    print("📌 הממשק ייפתח אוטומטית בדפדפן")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,  # יוצר לינק ציבורי
        show_error=True
    )
