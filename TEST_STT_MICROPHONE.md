# STT Microphone Test

Detta är ett fristående testskript för att testa STT (Speech-To-Text) funktionalitet med mikrofonen, isolerat från övriga moduler (wake word, VAD, MQTT, TTS).

## Syfte

Testet verifierar att:
- Vosk STT-modellen kan laddas korrekt
- Mikrofonen kan spela in ljud
- Inspelat ljud kan transkriberas till text
- STT-modulen fungerar som förväntat

## Förutsättningar

1. Python 3.9 eller senare
2. Installerade beroenden från `requirements.txt`
3. Nedladdad Vosk-modell (se [README.md](README.md) för instruktioner)
4. Fungerande mikrofon

## Installation

```bash
# Skapa och aktivera virtuell miljö
python3 -m venv venv
source venv/bin/activate

# Installera beroenden
pip install -r requirements.txt

# Ladda ner Vosk-modell (om inte redan gjort)
mkdir -p models/vosk
cd models/vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-sv-rhasspy-0.15.zip
unzip vosk-model-small-sv-rhasspy-0.15.zip
rm vosk-model-small-sv-rhasspy-0.15.zip
cd ../..
```

## Användning

### Grundläggande test

Kör testet med standardinställningar (5 sekunders inspelning):

```bash
python test_stt_microphone.py
```

### Spara och spela upp inspelat ljud

För att felsöka problem med STT kan du spara inspelningen och spela upp den:

```bash
# Spara inspelningen till en WAV-fil
python test_stt_microphone.py --save-audio

# Spara och spela upp inspelningen direkt
python test_stt_microphone.py --play-audio
```

När du använder `--save-audio` eller `--play-audio` sparas ljudfilen i `test_recordings/`-katalogen med ett tidsstämpel-baserat filnamn. Du kan sedan lyssna på filen för att avgöra om problemet ligger i:
- Mikrofonkvalitet eller volym
- Bakgrundsljud
- Talartydlighet
- Modellens förmåga att känna igen det talade språket

Du kan också spela upp den sparade filen manuellt med:
```bash
aplay test_recordings/test_YYYYMMDD_HHMMSS.wav
```

### Lista tillgängliga ljudenheter

För att se vilka mikrofoner som är tillgängliga:

```bash
python test_stt_microphone.py --list-devices
```

Detta visar alla tillgängliga ljudingångar med deras index-nummer.

### Testa med specifik mikrofon

Om du har flera mikrofoner kan du välja vilken som ska användas:

```bash
python test_stt_microphone.py --device 2
```

Ersätt `2` med index-numret från `--list-devices`.

### Anpassa inspelningstid

För att spela in längre eller kortare:

```bash
# 10 sekunders inspelning
python test_stt_microphone.py --duration 10

# 3 sekunders inspelning
python test_stt_microphone.py --duration 3
```

### Använd annan modell

Om du har laddat ner den större svenska modellen:

```bash
python test_stt_microphone.py --model-path models/vosk/vosk-model-sv-se-0.22
```

### Verbose logging

För detaljerad information om vad som händer:

```bash
python test_stt_microphone.py --verbose
```

## Exempel på utdata

### Lyckat test

```
======================================================================
  STT (Speech-To-Text) Microphone Test
======================================================================
  Model: models/vosk/vosk-model-small-sv-rhasspy-0.15
  Language: sv
  Sample Rate: 16000 Hz
  Recording Duration: 5 seconds
  Input Device: [default] USB Microphone
======================================================================

Initializing STT module...
✓ STT module initialized successfully

======================================================================
Recording for 5 seconds...
Please speak clearly into the microphone.
======================================================================

======================================================================
Recording complete!
======================================================================

Transcribing audio...

======================================================================
TRANSCRIPTION RESULT:
======================================================================
  "hej det här är ett test av rösttranskribering"
======================================================================

✓ STT test completed successfully!
```

### Varningar

Om ingen text transkriberas kan du få följande varning:

```
======================================================================
TRANSCRIPTION RESULT:
======================================================================
  (no text detected)
======================================================================

⚠ WARNING: No text was transcribed
  This could mean:
  - No speech was detected in the audio
  - The microphone volume is too low
  - Background noise is too high
  - The language model doesn't match the spoken language
```

## Felsökning

### Problem: "No module named 'numpy'"

Du behöver installera beroendena:

```bash
pip install -r requirements.txt
```

### Problem: "Vosk model not found"

Kontrollera att du har laddat ner Vosk-modellen till rätt plats. Se [README.md](README.md) för instruktioner.

### Problem: "No text was transcribed"

1. **Kontrollera mikrofonvolymen**: 
   ```bash
   # Lista enheter och kontrollera standard-mikrofon
   python test_stt_microphone.py --list-devices
   ```

2. **Testa mikrofoninspelning**:
   ```bash
   # Spela in ett test
   arecord -d 5 -f cd test.wav
   # Lyssna på inspelningen
   aplay test.wav
   ```

3. **Justera mikrofonvolym**:
   ```bash
   alsamixer
   # Tryck F4 för "Capture" och justera nivån
   ```

4. **Prova längre inspelning**: 
   ```bash
   python test_stt_microphone.py --duration 10
   ```

5. **Tala tydligare**: Tala tydligt och nära mikrofonen, undvik bakgrundsljud

### Problem: Felaktig transkribering på svenska

1. **Använd större modell** för bättre noggrannhet:
   ```bash
   # Ladda ner större modell
   cd models/vosk
   wget https://alphacephei.com/vosk/models/vosk-model-sv-se-0.22.zip
   unzip vosk-model-sv-se-0.22.zip
   rm vosk-model-sv-se-0.22.zip
   cd ../..
   
   # Testa med större modell
   python test_stt_microphone.py --model-path models/vosk/vosk-model-sv-se-0.22
   ```

2. **Kontrollera ljudkvalitet**: Se till att mikrofonen är av god kvalitet och ordentligt ansluten

## Kommandoradsalternativ

```
usage: test_stt_microphone.py [-h] [--model-path MODEL_PATH]
                               [--sample-rate SAMPLE_RATE]
                               [--duration DURATION] [--device DEVICE]
                               [--language LANGUAGE] [--list-devices]
                               [--verbose] [--save-audio] [--play-audio]

Test STT (Speech-To-Text) functionality with microphone

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to Vosk model directory (default: from
                        config.yaml or models/vosk/vosk-model-small-sv-
                        rhasspy-0.15)
  --sample-rate SAMPLE_RATE
                        Sample rate in Hz (default: 16000)
  --duration DURATION   Recording duration in seconds (default: 5)
  --device DEVICE       Audio input device index (default: system default)
  --language LANGUAGE   Language code (default: sv for Swedish)
  --list-devices        List available audio devices and exit
  --verbose             Enable verbose debug logging
  --save-audio          Save the recorded audio to a WAV file for debugging
  --play-audio          Play back the recorded audio after transcription
```

## Integrering med huvudapplikationen

Detta testskript använder samma STT-modul (`modules/stt.py`) som huvudapplikationen (`main.py`), vilket säkerställer att testet verifierar den faktiska funktionalitet som används i produktionen.

Konfigurationsfilen (`config.yaml`) läses automatiskt om den finns, så samma modellsökväg och inställningar används som i huvudapplikationen.

## Licens

MIT License - se LICENSE-filen för detaljer.
