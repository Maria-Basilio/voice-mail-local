import re
from pathlib import Path

import pyttsx3
import sounddevice as sd
import torch
import whisper
from scipy.io.wavfile import write
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MESSAGES_DIR = Path("messages")
OUTPUT_DIR = Path("output")

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

WHISPER_MODEL_NAME = "base"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

SOURCE_LANGUAGE = "por_Latn"

TRANSLATION_MODELS = {
    "en": {
        "name": "Inglês",
        "language_code": "eng_Latn",
    },
    "es": {
        "name": "Espanhol",
        "language_code": "spa_Latn",
    },
}

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".mp4"}

# CACHE DOS MODELOS

whisper_model = None
translation_tokenizer = None
translation_model = None


def get_whisper_model():
    """
    Carrega o Whisper somente uma vez durante a execução.

    Na primeira chamada, o modelo poderá ser baixado.
    Nas chamadas seguintes, o mesmo modelo em memória é reutilizado.
    """
    global whisper_model

    if whisper_model is None:
        print("\nCarregando modelo Whisper...")
        print(f"Modelo: {WHISPER_MODEL_NAME}")

        try:
            whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

        except Exception as exc:
            whisper_model = None

            raise RuntimeError(
                "Não foi possível carregar o modelo Whisper. "
                "Verifique a instalação do Whisper/PyTorch, "
                "a conexão com a internet na primeira execução "
                "e o espaço disponível em disco."
            ) from exc

    return whisper_model


def get_translation_model():
    """
    Carrega tokenizer e modelo NLLB somente uma vez durante a execução.

    O Hugging Face utiliza cache local para evitar novo download
    nas execuções seguintes.
    """
    global translation_tokenizer
    global translation_model

    if translation_tokenizer is None or translation_model is None:
        print("\nCarregando modelo de tradução local...")
        print(f"Modelo: {TRANSLATION_MODEL_NAME}")
        print(
            "Na primeira execução, o modelo poderá ser baixado "
            "do Hugging Face."
        )

        try:
            translation_tokenizer = AutoTokenizer.from_pretrained(
                TRANSLATION_MODEL_NAME
            )

            translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATION_MODEL_NAME
            )

            translation_model.eval()

        except Exception as exc:
            translation_tokenizer = None
            translation_model = None

            raise RuntimeError(
                "Não foi possível carregar o modelo de tradução local. "
                "Na primeira execução é necessário ter conexão com "
                "a internet para baixar o modelo e espaço suficiente "
                "em disco."
            ) from exc

    return translation_tokenizer, translation_model


# SAUDAÇÃO

def synthesize_greeting():
    """Converte texto digitado pelo usuário em um arquivo WAV."""

    text = input("\nDigite a mensagem da saudação: ").strip()

    if not text:
        print("\nTexto vazio. Saudação não criada.")
        return

    engine = None

    try:
        engine = pyttsx3.init()

        print("\nGerando saudação...")

        engine.save_to_file(
            text,
            str(OUTPUT_DIR / "greeting.wav")
        )

        engine.runAndWait()

        print("Saudação criada em: output/greeting.wav")

    except Exception as exc:
        print(f"\nErro ao gerar saudação: {exc}")

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


# WHISPER

def transcribe_audio(file_path, model=None):
    """
    Transcreve um arquivo de áudio utilizando Whisper.

    A transcrição é forçada para português.
    """

    if model is None:
        model = get_whisper_model()

    print(f"\nTranscrevendo: {file_path}")

    result = model.transcribe(
        str(file_path),
        language="pt",
        task="transcribe",
        fp16=False
    )

    text = result.get("text", "").strip()

    print("\nTranscrição:")
    print(text if text else "[texto vazio]")

    return text


# GRAVAÇÃO

def record_message():
    """Grava aproximadamente 5 segundos pelo microfone."""

    output_file = MESSAGES_DIR / "my_message.wav"

    print("\nGravando por 5 segundos...")
    print("Fale agora!")

    try:
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        write(
            str(output_file),
            SAMPLE_RATE,
            audio
        )

        print("\nGravação concluída!")
        print(f"Mensagem salva em: {output_file}")

    except Exception as exc:
        print(f"\nErro ao acessar/gravar pelo microfone: {exc}")


# ARQUIVOS

def get_audio_files():
    """Retorna os arquivos de áudio existentes em messages/."""

    if not MESSAGES_DIR.exists():
        return []

    return sorted(
        file
        for file in MESSAGES_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in AUDIO_EXTENSIONS
    )


def get_transcription_files():
    """
    Retorna somente as transcrições originais em português.

    Arquivos *_en.txt e *_es.txt são ignorados.
    """

    if not OUTPUT_DIR.exists():
        return []

    return sorted(
        file
        for file in OUTPUT_DIR.glob("*.txt")
        if file.is_file()
        and not re.search(r"_(en|es)$", file.stem)
    )


def save_text(text, output_file):
    """Salva texto utilizando UTF-8."""

    output_file.write_text(
        text,
        encoding="utf-8"
    )


# OPÇÃO 3

def transcribe_messages():
    """
    Opção 3.

    Transcreve todos os arquivos de áudio da pasta messages/
    e salva somente as transcrições em português.

    Não realiza tradução.
    Não gera áudio.
    """

    audio_files = get_audio_files()

    if not audio_files:
        print("\nNenhuma mensagem encontrada em messages/.")
        return

    try:
        model = get_whisper_model()

    except Exception as exc:
        print(f"\nErro ao carregar Whisper: {exc}")
        return

    print("\n" + "=" * 50)
    print("TRANSCRIÇÃO DAS MENSAGENS")
    print("=" * 50)

    for audio_file in audio_files:
        print("\n" + "-" * 50)
        print(f"Arquivo: {audio_file.name}")

        try:
            text = transcribe_audio(
                audio_file,
                model
            )

            output_file = (
                OUTPUT_DIR / f"{audio_file.stem}.txt"
            )

            save_text(
                text,
                output_file
            )

            print(f"\nTranscrição salva em: {output_file}")

        except Exception as exc:
            print(
                f"\nErro ao transcrever "
                f"{audio_file.name}: {exc}"
            )

    print("\nTranscrição concluída.")


# NLLB

def split_text(text, max_chars=1800):
    """
    Divide textos longos em partes menores.

    Isso evita que uma transcrição grande seja simplesmente
    truncada pelo limite de contexto do modelo.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= max_chars:
            current = candidate

        else:
            if current:
                chunks.append(current)

            # Se uma única frase for muito grande,
            # divide por tamanho.
            if len(sentence) > max_chars:
                for start in range(
                    0,
                    len(sentence),
                    max_chars
                ):
                    chunks.append(
                        sentence[start:start + max_chars]
                    )

                current = ""

            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def translate_text(text, target_language):
    """
    Traduz texto em português para inglês ou espanhol
    utilizando NLLB local.

    Nenhuma API de tradução é utilizada.
    """

    if not text or not text.strip():
        return ""

    if target_language not in TRANSLATION_MODELS:
        raise ValueError(
            f"Idioma de destino não suportado: {target_language}"
        )

    tokenizer, model = get_translation_model()

    target_code = TRANSLATION_MODELS[
        target_language
    ]["language_code"]

    tokenizer.src_lang = SOURCE_LANGUAGE

    chunks = split_text(text)

    if not chunks:
        return ""

    translated_chunks = []

    for index, chunk in enumerate(chunks, start=1):

        if len(chunks) > 1:
            print(
                f"Traduzindo parte "
                f"{index}/{len(chunks)}..."
            )

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        forced_bos_token_id = (
            tokenizer.convert_tokens_to_ids(
                target_code
            )
        )

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )

        translated = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()

        if translated:
            translated_chunks.append(translated)

    return "\n\n".join(translated_chunks).strip()


# VOZES / PYTTSX3

def find_voice_for_language(engine, language_code):
    """
    Procura uma voz instalada compatível com o idioma.

    Retorna True quando encontra uma voz compatível.
    """

    language_aliases = {
        "en": [
            "en",
            "english",
            "ingl",
            "eng",
            "en-us",
            "en_us",
            "en-gb",
            "en_gb",
        ],
        "es": [
            "es",
            "spanish",
            "espanol",
            "español",
            "spa",
            "spa_latn",
        ],
    }

    aliases = language_aliases.get(
        language_code,
        []
    )

    try:
        voices = engine.getProperty("voices")

        for voice in voices:
            values = [
                getattr(voice, "id", ""),
                getattr(voice, "name", ""),
                getattr(voice, "languages", ""),
            ]

            voice_text = " ".join(
                str(value).lower()
                for value in values
            )

            if any(
                alias.lower() in voice_text
                for alias in aliases
            ):
                engine.setProperty(
                    "voice",
                    voice.id
                )

                return True

    except Exception:
        pass

    return False


def synthesize_translation(
    text,
    target_language,
    output_file
):
    """
    Gera o áudio da tradução utilizando pyttsx3.

    Caso não exista uma voz específica para o idioma,
    utiliza a voz padrão disponível no sistema.
    """

    if not text or not text.strip():
        print(
            f"Texto vazio. Áudio em "
            f"{TRANSLATION_MODELS[target_language]['name']} "
            "não será criado."
        )

        return False

    engine = None

    try:
        engine = pyttsx3.init()

        voice_found = find_voice_for_language(
            engine,
            target_language
        )

        if voice_found:
            print(
                f"Voz encontrada para "
                f"{TRANSLATION_MODELS[target_language]['name']}."
            )

        else:
            print(
                f"Aviso: voz específica para "
                f"{TRANSLATION_MODELS[target_language]['name']} "
                "não encontrada."
            )

            print(
                "Usando a voz padrão disponível no sistema."
            )

        engine.save_to_file(
            text,
            str(output_file)
        )

        engine.runAndWait()

        return True

    except Exception as exc:
        print(
            f"Erro ao gerar áudio em "
            f"{TRANSLATION_MODELS[target_language]['name']}: "
            f"{exc}"
        )

        return False

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


# TRADUÇÃO DE UMA TRANSCRIÇÃO

def translate_transcription_file(
    transcription_file,
    generate_audio=False
):
    """
    Traduz uma transcrição existente.

    generate_audio=False:
        gera somente TXT EN/ES.

    generate_audio=True:
        gera TXT EN/ES e WAV EN/ES.
    """

    if not transcription_file.exists():
        print("\nA mensagem ainda não foi transcrita.")
        print(
            "Execute primeiro a opção 3 ou a opção 5."
        )

        return False

    text = transcription_file.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        print(
            f"\nA transcrição "
            f"{transcription_file.name} está vazia."
        )

        return False

    print("\n" + "=" * 50)
    print(
        f"TRADUZINDO: {transcription_file.name}"
    )
    print("=" * 50)

    success = False

    for target_language, config in TRANSLATION_MODELS.items():

        try:
            translated = translate_text(
                text,
                target_language
            )

            if not translated:
                print(
                    f"\nErro ao traduzir para "
                    f"{config['name']}: resultado vazio."
                )

                continue

            translated_file = (
                OUTPUT_DIR
                / f"{transcription_file.stem}_"
                f"{target_language}.txt"
            )

            save_text(
                translated,
                translated_file
            )

            print(f"\n{config['name']}:")
            print(translated)

            print(
                f"Texto salvo em: {translated_file}"
            )

            success = True

            # A opção 4 não entra aqui.
            # Somente o fluxo completo da opção 5
            # gera os arquivos WAV.
            if generate_audio:

                audio_file = (
                    OUTPUT_DIR
                    / f"{transcription_file.stem}_"
                    f"{target_language}.wav"
                )

                if synthesize_translation(
                    translated,
                    target_language,
                    audio_file
                ):
                    print(
                        f"Áudio salvo em: {audio_file}"
                    )

        except Exception as exc:
            print(
                f"\nErro ao traduzir para "
                f"{config['name']}: {exc}"
            )

            print(
                "Continuando para o próximo idioma..."
            )

    return success


# OPÇÃO 4

def translate_messages():
    """
    Opção 4.

    Traduz as transcrições existentes em output/.

    Não executa Whisper.
    Não gera WAV.
    """

    transcription_files = get_transcription_files()

    if not transcription_files:
        print(
            "\nA mensagem ainda não foi transcrita."
        )

        print(
            "Execute primeiro a opção 3 ou a opção 5."
        )

        return

    print("\n" + "=" * 50)
    print("TRADUÇÃO DAS TRANSCRIÇÕES")
    print("=" * 50)

    for transcription_file in transcription_files:

        translate_transcription_file(
            transcription_file,
            generate_audio=False
        )

    print("\nTradução concluída.")
    print(
        "Os arquivos WAV não foram gerados "
        "pela opção 4."
    )


# OPÇÃO 5

def transcribe_and_translate_messages():
    """
    Opção 5.

    Executa:

    messages/*.wav
        ↓
    Whisper
        ↓
    português TXT
        ↓
    NLLB
        ↓
    inglês/espanhol TXT
        ↓
    pyttsx3
        ↓
    inglês/espanhol WAV
    """

    audio_files = get_audio_files()

    if not audio_files:
        print(
            "\nNenhuma mensagem encontrada em messages/."
        )

        return

    try:
        model = get_whisper_model()

    except Exception as exc:
        print(
            f"\nErro ao carregar Whisper: {exc}"
        )

        return

    print("\n" + "=" * 50)
    print("TRANSCRIÇÃO E TRADUÇÃO")
    print("=" * 50)

    for audio_file in audio_files:

        print("\n" + "=" * 50)
        print(
            f"PROCESSANDO: {audio_file.name}"
        )
        print("=" * 50)

        try:
            
            # 1. Whisper
          
            text = transcribe_audio(
                audio_file,
                model
            )

            transcription_file = (
                OUTPUT_DIR
                / f"{audio_file.stem}.txt"
            )

            save_text(
                text,
                transcription_file
            )

            print(
                f"\nTranscrição salva em: "
                f"{transcription_file}"
            )

            if not text:
                print(
                    "\nTexto vazio."
                )

                print(
                    "Traduções e áudios não serão "
                    "executados para esta mensagem."
                )

                continue

            # 2. NLLB + TXT + WAV
            
            translate_transcription_file(
                transcription_file,
                generate_audio=True
            )

        except Exception as exc:
            print(
                f"\nErro ao processar "
                f"{audio_file.name}: {exc}"
            )

    print("\n" + "=" * 50)
    print("PROCESSO COMPLETO FINALIZADO")
    print("=" * 50)

# MENU

def main():

    MESSAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    while True:

        print("\n")
        print("=" * 40)
        print("       VOICE MAIL LOCAL")
        print("=" * 40)
        print()

        print("1 - Criar saudação")
        print("2 - Gravar mensagem")
        print("3 - Transcrever mensagens")
        print("4 - Traduzir mensagens")
        print("5 - Transcrever e traduzir")
        print("6 - Sair")

        option = input(
            "\nEscolha uma opção: "
        ).strip()

        if option == "1":

            synthesize_greeting()

        elif option == "2":

            record_message()

        elif option == "3":

            transcribe_messages()

        elif option == "4":

            translate_messages()

        elif option == "5":

            transcribe_and_translate_messages()

        elif option == "6":

            print(
                "\nEncerrando o programa."
            )

            break

        else:

            print(
                "\nOpção inválida."
            )


if __name__ == "__main__":
    main()