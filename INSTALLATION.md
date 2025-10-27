# Installation Guide - Voice Assistant för Raspberry Pi 5

Denna guide hjälper dig att installera och konfigurera den lokala röststyrda AI-assistenten på din Raspberry Pi 5.

## 📦 Innehållsförteckning

1. [Förutsättningar](#förutsättningar)
2. [Hårdvaruinställning](#hårdvaruinställning)
3. [Mjukvaruinstallation](#mjukvaruinstallation)
4. [Konfigurera tjänster](#konfigurera-tjänster)
5. [Första körningen](#första-körningen)
6. [Felsökning](#felsökning)

## 🔧 Förutsättningar

### Hårdvara
- **Raspberry Pi 5** (4GB eller 8GB RAM rekommenderat)
- **MicroSD-kort** (32GB eller större)
- **USB-mikrofon** eller HAT med inbyggd mikrofon
- **Högtalare** (via USB, 3.5mm jack, eller HDMI)
- **Strömlåda för Pi 5** (27W rekommenderat)
- **Internetanslutning** (WiFi eller Ethernet)

### Mjukvara
- Raspberry Pi OS (64-bit) - Bookworm eller senare
- Tillgång till terminal/SSH

## 🔌 Hårdvaruinställning

### 1. Installera Raspberry Pi OS

Om du inte redan har:

```bash
# Använd Raspberry Pi Imager
# Välj: Raspberry Pi OS (64-bit)
# Skriv till SD-kort
# Konfigurera WiFi och SSH via Imager-inställningarna
```

### 2. Anslut hårdvara

1. **Mikrofon**: Anslut USB-mikrofonen till en USB-port
2. **Högtalare**: Anslut högtalare via USB, 3.5mm eller HDMI
3. **Ström**: Anslut strömkabeln och starta Pi:n

### 3. Verifiera ljudenheter

```bash
# Lista alla ljudenheter
arecord -l    # Inspelningsenheter
aplay -l      # Uppspelningsenheter

# Testa mikrofon
arecord -d 5 -f cd test.wav
aplay test.wav

# Justera volym
alsamixer
```

## 💿 Mjukvaruinstallation

### 1. Uppdatera systemet

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Klona repositoryt

```bash
cd ~
git clone https://github.com/fredrik-svg/ai.git
cd ai
```

### 3. Kör installationsskriptet

```bash
chmod +x setup.sh
./setup.sh
```

Skriptet kommer att:
- Installera systemberoenden
- Skapa Python virtuell miljö
- Installera Python-paket
- Ladda ner TTS-modeller
- Skapa konfigurationsfil

### 4. Manuell installation (om setup.sh misslyckas)

```bash
# Installera systemberoenden
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libopenblas-dev \
    ffmpeg \
    git

# Skapa virtuell miljö
python3 -m venv venv
source venv/bin/activate

# Installera Python-paket
pip install --upgrade pip
pip install -r requirements.txt

# Skapa kataloger
mkdir -p models/piper

# Kopiera konfiguration
cp config.example.yaml config.yaml
```

## 🔑 Konfigurera tjänster

### 1. Picovoice (Wake Word)

1. Gå till [Picovoice Console](https://console.picovoice.ai/)
2. Skapa ett gratis konto
3. Skapa ett nytt projekt
4. Kopiera din **Access Key**
5. Lägg till Access Key i `config.yaml`:

```yaml
wake_word:
  access_key: "DIN_ACCESS_KEY_HÄR"
```

### 2. HiveMQ Cloud (MQTT Broker)

#### Skapa gratis HiveMQ Cloud-instans

1. Gå till [HiveMQ Cloud](https://www.hivemq.com/mqtt-cloud-broker/)
2. Skapa ett gratis konto
3. Skapa en ny **Serverless** cluster (gratis tier)
4. Skapa MQTT-credentials:
   - Användarnamn
   - Lösenord
5. Notera cluster URL

#### Konfigurera MQTT i config.yaml

```yaml
mqtt:
  broker: "xxx-xxx.s1.eu.hivemq.cloud"  # Din cluster URL
  port: 8883
  username: "ditt_användarnamn"
  password: "ditt_lösenord"
  topic_send: "assistant/input"
  topic_receive: "assistant/output"
  use_tls: true
```

### 3. n8n Workflow

#### Installation av n8n (valfritt - kan köras på annan maskin)

```bash
# Installera Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Installera n8n
sudo npm install -g n8n

# Starta n8n
n8n
```

#### Skapa n8n Workflow

1. Öppna n8n webgränssnitt (http://localhost:5678)
2. Skapa nytt workflow
3. Lägg till **MQTT Trigger** node:
   - Broker: Din HiveMQ cluster
   - Topic: `assistant/input`
   - Credentials: Dina MQTT-credentials
4. Lägg till bearbetningsnoder (t.ex. OpenAI, API-anrop)
5. Lägg till **MQTT** send node:
   - Topic: `assistant/output`
6. Aktivera workflow

**Exempel workflow:**
```
MQTT Trigger (assistant/input)
  ↓
Set Node (formatera data)
  ↓
HTTP Request/OpenAI (generera svar)
  ↓
MQTT Send (assistant/output)
```

### 4. Ladda ner TTS-modeller

#### Svensk modell (standard)

```bash
cd models/piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-sv-se-nst-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-sv-se-nst-medium.onnx.json
cd ../..
```

#### Alternativa språk

För andra språk, besök: https://github.com/rhasspy/piper/releases

Uppdatera sedan `config.yaml`:
```yaml
tts:
  model_path: "models/piper/DIN_MODELL.onnx"
```

## 🚀 Första körningen

### 1. Redigera konfiguration

```bash
nano config.yaml
```

Viktiga inställningar:
- `wake_word.access_key` - Din Picovoice key
- `mqtt.broker` - HiveMQ cluster URL
- `mqtt.username` och `mqtt.password`
- `stt.model` - Whisper modell (base för Pi 5)

### 2. Testa komponenter

```bash
source venv/bin/activate

# Testa ljudenheter
python test_utils.py --test-audio

# Testa mikrofon
python test_utils.py --test-mic

# Testa MQTT
python test_utils.py --test-mqtt

# Testa TTS
python test_utils.py --test-tts
```

### 3. Starta assistenten

```bash
source venv/bin/activate
python main.py
```

### 4. Använd assistenten

1. Vänta på meddelandet: "Listening for wake word..."
2. Säg wake word (standard: "jarvis" eller vad du konfigurerat)
3. När systemet svarar med ljud, säg din fråga
4. Assistenten skickar till n8n och spelar upp svaret

## 🐛 Felsökning

### Problem: "No audio devices found"

**Lösning:**
```bash
# Installera ALSA-verktyg
sudo apt-get install alsa-utils

# Lista enheter
aplay -l
arecord -l

# Konfigurera default enhet i ~/.asoundrc
```

### Problem: "Failed to connect to MQTT broker"

**Lösning:**
```bash
# Verifiera internetanslutning
ping 8.8.8.8

# Testa MQTT med mosquitto
sudo apt-get install mosquitto-clients
mosquitto_sub -h xxx.hivemq.cloud -p 8883 -t test --capath /etc/ssl/certs/ -u username -P password

# Kontrollera firewall
sudo ufw status
```

### Problem: "Out of memory" eller systemet fryser

**Lösning:**
```bash
# Använd mindre Whisper-modell
# I config.yaml:
stt:
  model: "tiny"  # eller "base"

# Öka swap-storlek
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Sätt CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Problem: "Wake word not detected"

**Lösning:**
```bash
# Kontrollera Picovoice Access Key
# Testa mikrofon-nivå
alsamixer  # Öka capture-nivå

# Använd ett built-in keyword
# I config.yaml:
wake_word:
  keyword: "jarvis"  # eller "computer", "pico clock", etc.
```

### Problem: TTS-fel

**Lösning:**
```bash
# Installera piper manuellt
pip install piper-tts

# Verifiera modellsökväg
ls -la models/piper/

# Testa TTS direkt
echo "test" | piper --model models/piper/voice-sv-se-nst-medium.onnx --output-file test.wav
```

## 🔄 Kör vid uppstart (valfritt)

För att automatiskt starta assistenten vid boot:

```bash
# Skapa systemd service
sudo nano /etc/systemd/system/voice-assistant.service
```

Innehåll:
```ini
[Unit]
Description=Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ai
Environment="PATH=/home/pi/ai/venv/bin"
ExecStart=/home/pi/ai/venv/bin/python /home/pi/ai/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Aktivera service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-assistant.service
sudo systemctl start voice-assistant.service

# Kontrollera status
sudo systemctl status voice-assistant.service

# Visa loggar
sudo journalctl -u voice-assistant.service -f
```

## 📚 Nästa steg

- Anpassa n8n workflows för dina behov
- Träna custom wake word på Picovoice Console
- Optimera Whisper-modellstorleken för balans mellan hastighet och noggrannhet
- Integrera med Home Assistant eller andra hemautomationssystem

## 🆘 Support

För hjälp, skapa en issue på GitHub-repositoryt.
