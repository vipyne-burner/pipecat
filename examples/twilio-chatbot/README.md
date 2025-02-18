# signalwire / websockets / gemini multimodal (text modality) / tts

This is an alternate version of the Twilio Chatbot.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configure Twilio URLs](#configure-twilio-urls)
- [Running the Application](#running-the-application)
- [Usage](#usage)

## Features

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+.
- **WebSocket Support**: Real-time communication using WebSockets.
- **CORS Middleware**: Allowing cross-origin requests for testing.
- **Dockerized**: Easily deployable using Docker.

## Requirements

- Python 3.10
- Docker (for containerized deployment)
- ngrok (for tunneling)
- Twilio Account

## Installation

1. **Set up a virtual environment** (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Create .env**:
   Copy the example environment file and update with your settings:

   ```sh
   cp env.example .env
   ```

4. **Install ngrok**:
   Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install ngrok.

## Configure Twilio URLs

1. **Start ngrok**:
   In a new terminal, start ngrok to tunnel the local server:

   ```sh
   ngrok http 8765
   ```

2. **Update the Signalwire Webhook**:

   - Go to Signalwire's cXML / LaML configuration page
     - Choose "Bins" submenu
     - Create a LaML Bin with your ngrok URL (e.g., https://<ngrok_url>) like so:
```
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Redirect>https://abc123.ngrok.app</Redirect>
</Response>
```
     - Click Save at the bottom of the page

   - Go to your Signalwire phone number's configuration page
   - Click on "Edit Settings"
   - Under "Inbound Call Settings"
     - Under the "Accept Incoming Calls As" section:
       - Select "Voice Calls" from the dropdown
     - Under the "Handle Calls Using" section:
       - Select "LaML Webhooks" from the dropdown
     - Under the "When a call comes in" section:
       - Select the LaML Bin you created with your ngrok url
:


3. **Configure streams.xml**:
   - Copy the template file to create your local version:
     ```sh
     cp templates/streams.xml.template templates/streams.xml
     ```
   - In `templates/streams.xml`, replace `<your server url>` with your ngrok URL (without `https://`)
   - The final URL should look like: `wss://abc123.ngrok.app/ws`

## Running the Application

Choose one of these two methods to run the application:

### Using Python (Option 1)

**Run the FastAPI application**:

```sh
# Make sure you’re in the project directory and your virtual environment is activated
python server.py
```

## Usage

Call to your configured Signalwire phone number. The webhook URL will direct the call to your FastAPI application, which will handle it accordingly.

