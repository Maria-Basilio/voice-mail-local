# Voice Mail Local

Assistente de mensagens de voz desenvolvido em Python, utilizando processamento
local de áudio para conversão de texto em fala e fala em texto.

## Objetivo

O projeto implementa um sistema simples de Voice Mail capaz de:

- Criar uma saudação em áudio a partir de texto
- Gravar uma mensagem pelo microfone
- Salvar a mensagem em formato WAV
- Transcrever mensagens de áudio utilizando Whisper
- Transcrever vários arquivos da pasta `messages`
- Funcionar sem serviços de nuvem

## Tecnologias

- Python 3.11 / 3.12
- OpenAI Whisper
- pyttsx3
- sounddevice
- scipy
- FFmpeg

## Estrutura

```text
voice-mail-local/
├── messages/
├── output/
├── .gitignore
└── voice_mail.py
