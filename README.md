# DeepSeek Memory

DeepSeek Memory is a Python-based interactive terminal application that augments a local Large Language Model (LLM) with persistent memory. It leverages **Ollama** to run a local LLM (e.g., **DeepSeek-R1:1.5b** or any other model you choose) and maintains conversation history across sessions. This enables personalized, context-aware interactions and allows the AI to recall details like the user's name and prior context over time.

---

## Features

- **Persistent Memory:**  
  Conversation history is saved to disk (in `dnc_memory.pt`), ensuring that user details and context are preserved across sessions.

- **Dynamic Context Handling:**  
  Uses embeddings and usage-based memory management to store and retrieve the most relevant conversation entries.

- **Customizable LLM Model:**  
  Easily switch the underlying LLM by modifying the `MODEL_NAME` variable in the code.

- **Special Query Handling:**  
  Recognizes and responds to queries like “What’s my name?” or “What’s your name?” using built-in heuristics.

- **Interactive Terminal Interface:**  
  Engage in natural, real-time conversation with the AI directly via the command line.

---

## Setup

### 1. Clone or Download the Repository

```bash
git clone https://github.com/yourusername/deepseek-memory.git
cd deepseek-memory
```

### 2. Ensure Your Environment Is Set Up

#### Python:
Install Python 3.8 or higher.

#### Ollama & Model:
Install Ollama and verify your chosen model works by running:

```bash
ollama run deepseek-r1:1.5b
```

To change the model used by the application, edit the `MODEL_NAME` variable in the code.

#### Dependencies:
Install the required Python packages:

```bash
pip install torch torchvision torchaudio numpy regex
```

### 3. Configure the Model

In the source code (`main.py`), adjust these global variables as needed:

- `MODEL_NAME`: Set to the name of the model (default: "deepseek-r1:1.5b").
- `BOT_NAME`: The bot’s name (default: "Jarvis").

---

## Usage

### Run the Program
Execute the script from your terminal:

```bash
python main.py
```

### Interaction Flow

#### Startup:
If no user name is stored, the bot (Jarvis) will prompt:

```
"My name is Jarvis. What's your name?"
```

The user's response is processed to extract and store only the name (e.g., "Rasim").

#### Conversation:
- Ask any questions in natural language.
- Special queries like “What’s my name?” or “What’s your name?” are handled by the system.
- For general queries, the recent conversation history is used as context to generate concise, direct answers.

#### Persistent Memory:
After each interaction, the conversation history is saved to disk (in `dnc_memory.pt`), ensuring context is preserved across sessions.

---

## How It Works

### 1. Memory Management

#### Input Conversion:
User inputs are converted into embeddings via the local LLM (using Ollama). If the LLM call fails, a fallback deterministic embedding (using SHA‑256) is used.

#### Storage:
A fixed-size memory stores these embeddings along with raw conversation text.

#### Usage Vector:
A usage vector decides which memory slot to update when new entries are added.

### 2. Special Query Handling

#### User Name Retrieval:
Queries like “What’s my name?” trigger a scan of the conversation history to retrieve the stored user name.

#### Bot Name Response:
The bot’s name is hard-coded (default: Jarvis) and is used when the query “What’s your name?” is asked.

### 3. Response Generation

For general queries, the conversation history (last five entries) is aggregated and sent to the LLM as context, which then generates a direct, concise answer.

---

## Troubleshooting

### PSReadline Warning
On Windows, you may see a PSReadline warning. This can be safely ignored.

### Ollama/Model Errors
If the local LLM call fails:

- Ensure Ollama is installed and the model runs correctly:

  ```bash
  ollama run deepseek-r1:1.5b
  ```

- Verify your model configuration and that the `MODEL_NAME` variable is set correctly.

### Memory Persistence Issues
Ensure that the directory where `dnc_memory.pt` is saved is writable.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
