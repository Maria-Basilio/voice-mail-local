# Voice Mail Local

Assistente de mensagens de voz desenvolvido em Python para gravação,
transcrição, tradução e geração de áudio utilizando processamento local.

O projeto permite gravar mensagens pelo microfone, converter áudio em texto
com Whisper, traduzir as transcrições para inglês e espanhol utilizando
NLLB-200 e gerar novos arquivos de áudio com as traduções.

A proposta principal é realizar o processamento localmente, sem utilização de
APIs pagas de tradução ou serviços de tradução em nuvem.

---

## Funcionalidades

O projeto implementa:

- Criar uma saudação em áudio a partir de texto
- Gravar mensagens pelo microfone
- Salvar mensagens em formato WAV
- Transcrever mensagens de áudio utilizando Whisper
- Transcrever vários arquivos da pasta `messages`
- Traduzir transcrições para inglês
- Traduzir transcrições para espanhol
- Gerar arquivos TXT com as traduções
- Gerar arquivos WAV com as traduções
- Utilizar processamento local para tradução
- Reutilizar modelos através do cache local
- Processar várias mensagens automaticamente

---

## Tecnologias utilizadas

- Python 3.12
- OpenAI Whisper
- PyTorch
- Hugging Face Transformers
- NLLB-200
- SentencePiece
- pyttsx3
- sounddevice
- SciPy
- FFmpeg

---

## Funcionamento

O fluxo principal do sistema é:

```text
Microfone
    |
    v
Arquivo WAV
    |
    v
Whisper
    |
    v
Transcrição em português
    |
    v
NLLB-200
    |
    +-------------------+
    |                   |
    v                   v
Inglês              Espanhol
    |                   |
    v                   v
Arquivo TXT         Arquivo TXT
    |                   |
    v                   v
Arquivo WAV         Arquivo WAV