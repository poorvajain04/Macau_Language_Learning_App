import os
import json
from groq import Groq
from gtts import gTTS
from faster_whisper import WhisperModel
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
SUPPORTED_LANGUAGES = {
    "English": {
        "tts": "en",
        "whisper": "en",
        "native": "English"
    },
    "Hindi": {
        "tts": "hi",
        "whisper": "hi",
        "native": "हिन्दी"
    },
    "Spanish": {
        "tts": "es",
        "whisper": "es",
        "native": "Español"
    },
    "French": {
        "tts": "fr",
        "whisper": "fr",
        "native": "Français"
    },
    "German": {
        "tts": "de",
        "whisper": "de",
        "native": "Deutsch"
    },
    "Italian": {
        "tts": "it",
        "whisper": "it",
        "native": "Italiano"
    },
    "Portuguese": {
        "tts": "pt",
        "whisper": "pt",
        "native": "Português"
    },
    "Russian": {
        "tts": "ru",
        "whisper": "ru",
        "native": "Русский"
    },
    "Japanese": {
        "tts": "ja",
        "whisper": "ja",
        "native": "日本語"
    },
    "Korean": {
        "tts": "ko",
        "whisper": "ko",
        "native": "한국어"
    },
    "Chinese": {
        "tts": "zh-CN",
        "whisper": "zh",
        "native": "中文"
    },
    "Arabic": {
        "tts": "ar",
        "whisper": "ar",
        "native": "العربية"
    },
    "Bengali": {
        "tts": "bn",
        "whisper": "bn",
        "native": "বাংলা"
    },
    "Tamil": {
        "tts": "ta",
        "whisper": "ta",
        "native": "தமிழ்"
    },
    "Telugu": {
        "tts": "te",
        "whisper": "te",
        "native": "తెలుగు"
    },
    "Marathi": {
        "tts": "mr",
        "whisper": "mr",
        "native": "मराठी"
    },
    "Gujarati": {
        "tts": "gu",
        "whisper": "gu",
        "native": "ગુજરાતી"
    },
    "Punjabi": {
        "tts": "pa",
        "whisper": "pa",
        "native": "ਪੰਜਾਬੀ"
    },
    "Malayalam": {
        "tts": "ml",
        "whisper": "ml",
        "native": "മലയാളം"
    },
    "Kannada": {
        "tts": "kn",
        "whisper": "kn",
        "native": "ಕನ್ನಡ"
    },
    "Urdu": {
        "tts": "ur",
        "whisper": "ur",
        "native": "اردو"
    },
    "Turkish": {
        "tts": "tr",
        "whisper": "tr",
        "native": "Türkçe"
    },
    "Vietnamese": {
        "tts": "vi",
        "whisper": "vi",
        "native": "Tiếng Việt"
    },
    "Thai": {
        "tts": "th",
        "whisper": "th",
        "native": "ไทย"
    },
    "Dutch": {
        "tts": "nl",
        "whisper": "nl",
        "native": "Nederlands"
    },
    "Greek": {
        "tts": "el",
        "whisper": "el",
        "native": "Ελληνικά"
    },
    "Swedish": {
        "tts": "sv",
        "whisper": "sv",
        "native": "Svenska"
    },
    "Polish": {
        "tts": "pl",
        "whisper": "pl",
        "native": "Polski"
    },
    "Czech": {
        "tts": "cs",
        "whisper": "cs",
        "native": "Čeština"
    },
    "Romanian": {
        "tts": "ro",
        "whisper": "ro",
        "native": "Română"
    }
}
def text_to_speech(text, target_language):

    lang_code = SUPPORTED_LANGUAGES[target_language]["tts"]

    output_file = "reply.mp3"

    tts = gTTS(
        text=text,
        lang=lang_code,
        slow=False
    )

    tts.save(output_file)

    return output_file
def analyze_student(user_text, target_language):
    prompt = f"""
You are an expert language teacher.

Student originally said:
"{user_text}"

Target language: {target_language}

Your tasks:

1. Estimate the learner's CEFR level.
Choose ONLY one:
A1
A2
B1
B2
C1
C2

2. Translate the student's speech into natural {target_language}.

3. Reply as the tutor ONLY in {target_language}.
If needed:
- politely correct grammar
- appreciate the learner
- continue the conversation
- ask ONE follow-up question

Return ONLY valid JSON.

Example:

{{
"level":"B1",
"translated_student":"...",
"tutor_reply":"..."
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"Return ONLY JSON."},
            {"role":"user","content":prompt}
        ],
        temperature=0.3,
        response_format={"type":"json_object"}
    )

    return json.loads(response.choices[0].message.content)
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)
def speech_to_text(audio_path):

    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    text = ""

    for segment in segments:
        text += segment.text + " "

    return text, info.language
