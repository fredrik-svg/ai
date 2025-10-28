# 🎙️ Lokal Röststyrd AI-Assistent för Raspberry Pi 5

En helt lokal, röststyrd AI-assistent som körs på Raspberry Pi 5. Systemet kan lyssna, tolka och svara med tal, samt integrera med automatiseringsflöden i n8n via MQTT (HiveMQ Cloud).

## ✨ Funktioner

- **Lokal bearbetning**: All röstbearbetning sker lokalt på Pi:n
- **Wake Word Detection**: Aktiveras med "Hey Genio" (eller anpassat ord)
- **Voice Activity Detection (VAD)**: Intelligent detektering av tal
- **Speech-to-Text (STT)**: Lokal transkribering med Whisper/Faster-Whisper
- **MQTT Integration**: Kommunicerar med n8n via HiveMQ Cloud
- **Text-to-Speech (TTS)**: Lokal talsyntes med Piper
- **Privacy-first**: Ingen röstdata skickas till molnet

## 🧩 Arkitektur

Systemet består av fem huvudmoduler:

| Modul | Funktion | Teknologi |
|-------|----------|-----------|
| **Wake Word** | Detekterar aktiveringsord | Porcupine |
| **VAD** | Avgör när användaren talar | WebRTC VAD / Silero VAD |
| **STT** | Transkriberar tal till text | Faster-Whisper |
| **Dialog/MQTT** | Kommunicerar med n8n | Paho MQTT |
| **TTS** | Syntetiserar svar till tal | Piper |

## 📋 Förutsättningar

### Hårdvara
- Raspberry Pi 5 (rekommenderat 4GB+ RAM)
- USB-mikrofon eller HAT med mikrofon
- Högtalare (USB, 3.5mm eller HDMI-ljud)
- MicroSD-kort (32GB+ rekommenderat)

### Mjukvara
- Raspberry Pi OS (64-bit rekommenderat)
- Python 3.9 eller senare
- Internetuppkoppling för MQTT

## 🚀 Installation

### 1. Klona repositoryt
```bash
git clone https://github.com/fredrik-svg/ai.git
cd ai
```

### 2. Installera systemberoenden
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev libopenblas-dev
```

### 3. Skapa virtuell miljö
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Installera Python-paket
```bash
pip install -r requirements.txt
```

### 5. Ladda ner modeller

#### Porcupine Wake Word (kräver Access Key)
Registrera dig på [Picovoice Console](https://console.picovoice.ai/) för att få en gratis Access Key.

**För anpassade wake words:**
1. Gå till Picovoice Console och skapa ett anpassat wake word
2. Ladda ner `.ppn`-filen
3. Placera den i projektets katalog (t.ex. `models/wake_words/`)
4. Uppdatera `keyword_path` i `config.yaml` med sökvägen till din `.ppn`-fil

#### Faster-Whisper modell
```bash
# Modellen laddas ner automatiskt vid första körningen
# Rekommenderad modell för Pi 5: base eller small
```

#### Piper TTS modell
```bash
mkdir -p models/piper
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/lisa/medium/sv_SE-lisa-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/sv/sv_SE/lisa/medium/sv_SE-lisa-medium.onnx.json
cd ../..
```

### 6. Konfigurera
Kopiera exempel-konfigurationen och anpassa den:
```bash
cp config.example.yaml config.yaml
nano config.yaml
```

Redigera `config.yaml` och fyll i:
- Picovoice Access Key
- HiveMQ Cloud MQTT-inställningar (URL, port, användarnamn, lösenord)
- MQTT-topics för n8n-integration

## ⚙️ Konfiguration

Redigera `config.yaml`:

```yaml
# Wake Word Detection
wake_word:
  access_key: "DIN_PICOVOICE_ACCESS_KEY"
  keyword: "hey-genio"  # Built-in keyword (används bara om keyword_path inte är satt)
  keyword_path: null  # Sökväg till anpassad .ppn-fil (t.ex., "models/wake_words/genio.ppn")
  sensitivity: 0.5

# Voice Activity Detection
vad:
  sample_rate: 16000
  frame_duration: 30  # ms
  
# Speech to Text
stt:
  model: "base"  # tiny, base, small, medium, large
  language: "sv"  # svenska
  device: "cpu"
  beam_size: 8  # Strålstorlek för avkodning (högre = bättre kvalitet men långsammare)
  temperature: 0.0  # Temperatur för sampling (0.0 = deterministisk)
  initial_prompt: "Detta är en konversation på svenska."  # Ledtråd för bättre svensk igenkänning
  vad_filter: true  # Använd VAD för att filtrera bort tystnad
  vad_min_silence_duration: 700  # Minsta tystnadsvaraktighet i ms

# MQTT / n8n Integration
mqtt:
  broker: "YOUR_HIVEMQ_CLUSTER.hivemq.cloud"
  port: 8883
  username: "your_username"
  password: "your_password"
  topic_send: "assistant/input"
  topic_receive: "assistant/output"
  use_tls: true

# Text to Speech
tts:
  model_path: "models/piper/sv_SE-lisa-medium.onnx"
  config_path: "models/piper/sv_SE-lisa-medium.onnx.json"

# Audio Settings
audio:
  input_device: null  # null = default device
  output_device: null  # null = default device
  sample_rate: 16000
  channels: 1
```

## 🎯 Användning

### Starta assistenten
```bash
python main.py
```

### Interagera
1. Säg wake word: "Hey Genio"
2. Vänta på bekräftelseljud
3. Säg din fråga eller kommando
4. Assistenten svarar via högtalarna

### Exempel på interaktion
```
Användare: "Hey Genio"
System: *pling*
Användare: "Vad är klockan?"
Assistent: "Klockan är 14:30"
```

## 🔧 n8n Integration

### Setup i n8n
1. Skapa ett nytt workflow i n8n
2. Lägg till en **MQTT Trigger** node
   - Anslut till din HiveMQ Cloud broker
   - Lyssna på topic: `assistant/input`
3. Bearbeta inkommande text (t.ex. med AI, API-anrop, etc.)
4. Skicka svar via **MQTT** node
   - Skicka till topic: `assistant/output`

### Exempel n8n Workflow
```
MQTT Trigger (assistant/input)
  ↓
Function Node (bearbeta text)
  ↓
OpenAI/LLM Node (generera svar)
  ↓
MQTT Send (assistant/output)
```

## 📁 Projektstruktur

```
ai/
├── main.py                 # Huvudapplikation
├── config.yaml             # Konfiguration
├── config.example.yaml     # Exempel-konfiguration
├── requirements.txt        # Python-beroenden
├── README.md              # Denna fil
├── modules/
│   ├── __init__.py
│   ├── wake_word.py       # Porcupine wake word detection
│   ├── vad.py             # Voice Activity Detection
│   ├── stt.py             # Speech-to-Text (Faster-Whisper)
│   ├── mqtt_handler.py    # MQTT kommunikation
│   └── tts.py             # Text-to-Speech (Piper)
└── models/
    └── piper/             # TTS-modeller
```

## 🐛 Felsökning

### Mikrofonproblem
```bash
# Lista tillgängliga ljudenheter
python -c "import sounddevice as sd; print(sd.query_devices())"

# Testa mikrofon
arecord -l
arecord -d 5 test.wav
aplay test.wav
```

### Wake Word detektion är långsam eller missar aktivering
Om du har svårt att få respons på wake word:

1. **Justera känslighet**: Öka `sensitivity` värdet i `config.yaml` från 0.5 till 0.6-0.7
   ```yaml
   wake_word:
     sensitivity: 0.7  # Högre värde = mer responsiv
   ```

2. **Kontrollera mikrofonplacering**: Se till att mikrofonen är tillräckligt nära och inte blockerad

3. **Testa olika wake words**: Prova olika inbyggda wake words eller skapa ett anpassat

4. **Kontrollera systembelastning**: Kör `top` för att se om CPU är överbelastad

### STT (Speech-to-Text) kvalitetsproblem
Om rösttranskriptionen är dålig eller missar ord:

1. **Öka beam_size**: Högre värde ger bättre kvalitet men långsammare transkribering
   ```yaml
   stt:
     beam_size: 10  # Standard: 8, prova 10-12 för ännu bättre kvalitet
   ```

2. **Justera VAD-parametrar**: Om tal klipps av för tidigt
   ```yaml
   stt:
     vad_min_silence_duration: 1000  # Öka från 700ms till 1000ms
   ```

3. **Använd större modell**: Byt från "base" till "small" eller "medium"
   ```yaml
   stt:
     model: "small"  # Bättre noggrannhet men kräver mer minne och CPU
   ```

4. **Anpassa initial_prompt för svenska**: Lägg till vanliga svenska ord och fraser
   ```yaml
   stt:
     initial_prompt: "Detta är en konversation på svenska med vardagliga fraser och meningar."
   ```

5. **Aktivera condition_on_previous_text**: För bättre sammanhang mellan meningar
   ```yaml
   stt:
     condition_on_previous_text: true  # Använd tidigare text som kontext
   ```

6. **Kontrollera mikrofonkvalitet**: Testa med `arecord` och lyssna på inspelningen
   ```bash
   arecord -d 5 -f cd test.wav && aplay test.wav
   ```

**Tips för bästa svenska igenkänning**:
- Tala tydligt och i normal hastighet
- Undvik dialektala ord om möjligt, eller lägg till dem i initial_prompt
- För specialiserade domäner (t.ex. hemautomation), anpassa initial_prompt:
  ```yaml
  initial_prompt: "Detta är hemautomation kommandon på svenska. Tänd lampan. Stäng av värmen."
  ```

### ONNX Runtime GPU-varningar
Om du ser varningar om GPU-enheter som inte hittas (t.ex. "GPU device discovery failed"):
```
[W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] 
GPU device discovery failed: device_discovery.cc:89 ReadFileContents 
Failed to open file: "/sys/class/drm/card1/device/vendor"
```

Detta är normalt på CPU-baserade enheter som Raspberry Pi och kan ignoreras. 
Systemet är konfigurerat att undertrycka dessa varningar automatiskt, men de 
kan fortfarande visas vid första körningen. De påverkar inte funktionaliteten.

### MQTT-anslutningsproblem
- Kontrollera att HiveMQ Cloud-credentials är korrekta
- Verifiera att port 8883 är öppen i din firewall
- Testa anslutning med MQTT Explorer eller mosquitto_pub/sub

### Prestandaproblem
- Använd mindre Whisper-modell (tiny eller base)
- Överklockade Pi 5 om möjligt
- Säkerställ tillräcklig kylning

## 📝 Licens

MIT License - se LICENSE-filen för detaljer.

## 🤝 Bidrag

Pull requests är välkomna! För större ändringar, öppna gärna en issue först.

## 📧 Kontakt

För frågor eller support, skapa en issue i detta repository.
