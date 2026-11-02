# Voice Mail Local — Local Voice Messaging Assistant

<div align="center">

**Record. Transcribe. Translate. Listen.**

A Python-based voice messaging assistant powered by local speech recognition, machine translation, and audio generation.

</div>

---

## About the Project

**Voice Mail Local** is a Python application designed to record, transcribe, translate, and generate voice messages using locally executed processing models.

The application allows users to record audio messages through a microphone, convert speech into text using **OpenAI Whisper**, translate transcriptions into English and Spanish using **NLLB-200**, and generate new audio files containing the translated messages.

The main goal is to provide a local processing workflow without relying on paid translation APIs or cloud-based translation services.

By combining speech recognition, neural machine translation, and text-to-speech technologies, the project explores how different AI tools can work together in an automated voice messaging workflow.

## Features

- **Audio Greeting Generation:** Create an audio greeting from text.
- **Voice Recording:** Record messages directly through a microphone.
- **WAV Audio Storage:** Save recorded messages in WAV format.
- **Speech-to-Text:** Transcribe audio messages using Whisper.
- **Batch Transcription:** Process multiple audio files from the `messages` directory.
- **English Translation:** Translate Portuguese transcriptions into English.
- **Spanish Translation:** Translate Portuguese transcriptions into Spanish.
- **Text File Generation:** Save translated messages as TXT files.
- **Translated Audio Generation:** Generate WAV files containing translated messages.
- **Local Processing:** Perform translation locally without relying on cloud translation APIs.
- **Model Caching:** Reuse downloaded models through local caching.
- **Batch Processing:** Automatically process multiple voice messages.

## Technologies

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| OpenAI Whisper | Speech recognition and transcription |
| PyTorch | Machine learning framework |
| Hugging Face Transformers | Access to pretrained AI models |
| NLLB-200 | Neural machine translation |
| SentencePiece | Text tokenization |
| pyttsx3 | Text-to-speech generation |
| sounddevice | Audio recording through the microphone |
| SciPy | Audio processing and WAV file handling |
| FFmpeg | Audio format processing |

## How It Works

The application follows a voice processing pipeline that combines speech recognition, machine translation, and audio generation.

```text
             MICROPHONE
                  |
                  v
             WAV RECORDING
                  |
                  v
               WHISPER
                  |
                  v
       PORTUGUESE TRANSCRIPTION
                  |
                  v
               NLLB-200
                  |
          +-------+-------+
          |               |
          v               v
        ENGLISH         SPANISH
          |               |
          v               v
      TRANSLATED      TRANSLATED
         TEXT            TEXT
          |               |
          v               v
       TXT FILE        TXT FILE
          |               |
          v               v
       WAV AUDIO       WAV AUDIO
```

### Processing Workflow

1. **Recording:** The user records a voice message through the microphone.
2. **Audio Storage:** The recording is saved as a WAV file.
3. **Transcription:** Whisper converts the Portuguese audio into text.
4. **Translation:** NLLB-200 translates the transcription into English and Spanish.
5. **Text Generation:** The translated messages are saved as TXT files.
6. **Audio Generation:** The translated text is converted into new WAV audio files.

## Local AI Processing

One of the main aspects of Voice Mail Local is its focus on locally executed AI models.

The project uses Whisper for speech recognition and NLLB-200 for neural machine translation, allowing these processing stages to run locally after the required models have been downloaded.

This approach reduces dependence on external translation services and allows the application to process messages without sending their content to a cloud translation API.

The project also uses local model caching to avoid unnecessary downloads and improve subsequent executions.

## Project Goals

Voice Mail Local explores the integration of different AI technologies into a practical application.

Its main goal is to demonstrate how speech recognition, machine translation, and text-to-speech generation can be combined to create an automated multilingual voice messaging workflow.

The project also explores the advantages of local AI execution, particularly regarding independence from cloud translation services and the use of locally available models.

## Author

**Maria Eduarda Basilio**

Software Engineering Student | Python | Artificial Intelligence

[GitHub Profile](https://github.com/Maria-Basilio)

---

<div align="center">

**Voice Mail Local**

*Your voice, across languages. ♡*

</div>
