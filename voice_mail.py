import whisper
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path


MESSAGES_DIR = Path("messages")
OUTPUT_DIR = Path("output")

SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def synthesize_greeting():
    """Converte texto digitado pelo usuário em um arquivo WAV."""

    text = input("\nDigite a mensagem da saudação: ")

    engine = pyttsx3.init()

    print("\nGerando saudação...")

    engine.save_to_file(text, str(OUTPUT_DIR / "greeting.wav"))
    engine.runAndWait()

    print("Saudação criada em: output/greeting.wav")


def transcribe_audio(file_path):
    """Transcreve um arquivo de áudio usando Whisper."""

    print("\nCarregando modelo Whisper...")

    model = whisper.load_model("base")

    print(f"Transcrevendo: {file_path}")

    result = model.transcribe(
        str(file_path),
        language="pt",
        fp16=False
    )

    text = result["text"].strip()

    print("\nTranscrição:")
    print(text)

    return text


def record_message():
    """Grava aproximadamente 5 segundos pelo microfone."""

    print("\nGravando por 5 segundos...")
    print("Fale agora!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    output_file = MESSAGES_DIR / "my_message.wav"

    write(
        str(output_file),
        SAMPLE_RATE,
        audio
    )

    print("\nGravação concluída!")
    print(f"Mensagem salva em: {output_file}")


def transcribe_messages():
    """Transcreve todos os arquivos de áudio da pasta messages."""

    audio_extensions = [".wav", ".m4a", ".mp3", ".mp4"]

    audio_files = [
        file
        for file in MESSAGES_DIR.iterdir()
        if file.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print("\nNenhuma mensagem encontrada.")
        return

    print("\nCarregando modelo Whisper...")

    model = whisper.load_model("base")

    for audio_file in audio_files:

        print("\n" + "-" * 40)
        print(f"Arquivo: {audio_file.name}")

        result = model.transcribe(
            str(audio_file),
            language="pt",
            fp16=False
        )

        text = result["text"].strip()

        print(f"Transcrição: {text}")

        output_file = OUTPUT_DIR / f"{audio_file.stem}.txt"

        output_file.write_text(
            text,
            encoding="utf-8"
        )

        print(f"Salvo em: {output_file}")


def main():

    MESSAGES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    while True:

        print("\n")
        print("=== Voice Mail ===")
        print()
        print("1 - Criar saudação")
        print("2 - Gravar mensagem")
        print("3 - Transcrever mensagens")
        print("4 - Sair")

        option = input("\nEscolha uma opção: ")

        if option == "1":
            synthesize_greeting()

        elif option == "2":
            record_message()

        elif option == "3":
            transcribe_messages()

        elif option == "4":
            print("\nEncerrando o programa.")
            break

        else:
            print("\nOpção inválida.")


if __name__ == "__main__":
    main()
