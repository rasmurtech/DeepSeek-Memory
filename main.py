import os
import json
import subprocess
import torch
import torch.nn.functional as F
import hashlib
import numpy as np
import re

# Global variable for the model name (you can change this as needed)
MODEL_NAME = "deepseek-r1:1.5b"
BOT_NAME = "Jarvis"


def simple_text_embedding(text, dim=256):
    """
    Creates a deterministic embedding from text using SHA-256.
    This is a fallback embedding function.
    """
    h = hashlib.sha256(text.encode('utf-8')).digest()
    v = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    if len(v) < dim:
        v = np.pad(v, (0, dim - len(v)), mode='constant')
    else:
        v = v[:dim]
    return torch.tensor(v, dtype=torch.float)


def clean_response(response_text):
    """
    Removes chain-of-thought text enclosed in <think>...</think> tags,
    and strips extra whitespace.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    return cleaned.strip()


def extract_user_name(text):
    """
    Searches the given text for a pattern like "my name is <name>" (case-insensitive)
    and returns the extracted name if found.
    """
    pattern = re.compile(r"my name is\s+(\w+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def get_user_name_from_entries(entries, bot_name=BOT_NAME):
    """
    Scans the entries in reverse order and returns the most recent extracted name
    that is not equal to the bot's name.
    """
    for entry in reversed(entries):
        name = extract_user_name(entry)
        if name and name.lower() != bot_name.lower():
            return name
    return None


class PersistentDeepSeekMemory(torch.nn.Module):
    def __init__(self, memory_file, memory_size=10, embedding_dim=256):
        """
        Args:
            memory_file (str): Path to save/load persistent memory.
            memory_size (int): Number of memory slots.
            embedding_dim (int): Dimensionality of the embedding vector.
        """
        super(PersistentDeepSeekMemory, self).__init__()
        self.memory_file = memory_file
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        self.register_buffer('memory', torch.zeros(memory_size, embedding_dim))
        self.register_buffer('usage', torch.zeros(memory_size))
        self.entries = []

        if os.path.exists(self.memory_file):
            self.load_memory()
            print(f"Loaded memory from {self.memory_file}")
        else:
            print("Starting with a new memory.")

    def embed_text(self, text):
        """
        Calls Ollama to run the specified model to generate an embedding.
        If JSON parsing fails, uses a deterministic fallback embedding.
        """
        command = ["ollama", "run", MODEL_NAME, text]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace"
            )
            output_text = result.stdout.strip()
            if not output_text:
                print(
                    f"No output received from {MODEL_NAME} for text: '{text}'")
                response_text = text  # fallback to input text
            else:
                try:
                    output_json = json.loads(output_text)
                    if "embedding" in output_json:
                        embedding_list = output_json["embedding"]
                        if len(embedding_list) != self.embedding_dim:
                            if len(embedding_list) > self.embedding_dim:
                                embedding_list = embedding_list[:self.embedding_dim]
                            else:
                                embedding_list = embedding_list + \
                                    [0.0] * (self.embedding_dim -
                                             len(embedding_list))
                        return torch.tensor(embedding_list, dtype=torch.float)
                    elif "response" in output_json:
                        response_text = output_json["response"]
                    else:
                        response_text = output_text
                except json.JSONDecodeError:
                    response_text = output_text
            embedding_tensor = simple_text_embedding(
                response_text, self.embedding_dim)
        except subprocess.CalledProcessError as cpe:
            print(f"Error obtaining embedding for '{text}': {cpe}")
            print("stderr:", cpe.stderr)
            embedding_tensor = torch.zeros(
                self.embedding_dim, dtype=torch.float)
        except Exception as e:
            print(f"Error obtaining embedding for '{text}': {e}")
            embedding_tensor = torch.zeros(
                self.embedding_dim, dtype=torch.float)
        return embedding_tensor

    def write(self, text):
        """
        Stores a new memory entry using the generated embedding.
        Chooses the memory slot with the lowest usage.
        """
        embedding = self.embed_text(text)
        usage_scores = -self.usage  # lower usage gives higher score
        probabilities = F.softmax(usage_scores, dim=0)
        slot_index = torch.argmax(probabilities).item()
        self.memory[slot_index] = embedding
        self.usage.mul_(0.9)
        self.usage[slot_index] = 1.0

        if slot_index < len(self.entries):
            self.entries[slot_index] = text
        else:
            self.entries.append(text)
        return slot_index

    def read(self, query_text, top_k=3):
        """
        Retrieves memory entries similar to the query text.
        """
        query_embedding = self.embed_text(query_text)
        memory_norm = self.memory.norm(dim=1, keepdim=True) + 1e-10
        query_norm = query_embedding.norm() + 1e-10
        normalized_memory = self.memory / memory_norm
        normalized_query = query_embedding / query_norm
        cosine_sim = torch.matmul(normalized_memory, normalized_query)
        topk_values, topk_indices = torch.topk(cosine_sim, top_k)
        retrieved_entries = []
        for idx in topk_indices.tolist():
            if idx < len(self.entries):
                retrieved_entries.append(self.entries[idx])
            else:
                retrieved_entries.append("[Empty Slot]")
        return topk_indices, topk_values, retrieved_entries

    def save_memory(self):
        """
        Saves the current memory state to disk.
        """
        state = {"memory": self.memory,
                 "usage": self.usage, "entries": self.entries}
        torch.save(state, self.memory_file)
        print(f"Memory saved to {self.memory_file}")

    def load_memory(self):
        """
        Loads the memory state from disk.
        """
        state = torch.load(self.memory_file)
        self.memory.copy_(state["memory"])
        self.usage.copy_(state["usage"])
        self.entries = state["entries"]


def generate_response(prompt):
    """
    Calls Ollama to run the specified model to generate a conversational response.
    The prompt instructs the model to answer directly and concisely.
    """
    modified_prompt = (prompt +
                       "\nAnswer directly and concisely. " +
                       "Do not provide a summary of the conversation; " +
                       "just answer the user's question.")
    command = ["ollama", "run", MODEL_NAME, modified_prompt]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace"
        )
        output_text = result.stdout.strip()
        if not output_text:
            print(
                f"No output received from {MODEL_NAME} for prompt: '{prompt}'")
            return "[No response generated]"
        try:
            output_json = json.loads(output_text)
            response = output_json.get("response", "")
            if not response:
                response = output_text
        except json.JSONDecodeError:
            response = output_text
    except subprocess.CalledProcessError as cpe:
        print(f"Error generating response for prompt '{prompt}': {cpe}")
        print("stderr:", cpe.stderr)
        response = "[Error generating response]"
    except Exception as e:
        print(f"Error generating response for prompt '{prompt}': {e}")
        response = "[Error generating response]"

    return clean_response(response)


def main():
    memory_file = "dnc_memory.pt"
    embedding_dim = 256  # Dimensionality for embeddings.
    memory_size = 10     # Number of memory slots.

    memory_module = PersistentDeepSeekMemory(
        memory_file, memory_size, embedding_dim)
    print("\nWelcome to the AI terminal using DeepSeek. Type 'exit' to quit.")

    # Retrieve the user's name from memory (most recent non-bot name)
    stored_name = get_user_name_from_entries(
        memory_module.entries, bot_name=BOT_NAME)

    # If no name is stored, prompt the user.
    if not stored_name:
        initial_message = "My name is " + BOT_NAME + ". What's your name?"
        print("\nBot:", initial_message)
        memory_module.write(initial_message)
        user_input = input("\nYou: ").strip()
        # Process the input to extract only the name.
        extracted_name = extract_user_name(user_input)
        if not extracted_name:
            extracted_name = user_input  # fallback if regex doesn't match
        stored_name = extracted_name
        memory_module.write("My name is " + stored_name)
        print(f"[Memory updated with your name: {stored_name}]")
        print("\nBot: Nice to meet you, " + stored_name + "!")
        memory_module.save_memory()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break

        # Write the user's input into memory.
        slot = memory_module.write(user_input)
        print(f"[Memory updated in slot {slot}]")

        # Show the top 3 related memory entries.
        topk_indices, topk_values, retrieved_entries = memory_module.read(
            user_input, top_k=3)
        print("\nTop memory entries related to your input:")
        for i, entry in enumerate(retrieved_entries):
            print(
                f"  {i+1}. {entry} (similarity: {topk_values[i].item():.4f})")

        # Build conversation history from the last 5 entries.
        conversation_history = "\n".join(memory_module.entries[-5:])

        # Special query handling.
        if re.search(r"what(?:'s|s| is) my name\??", user_input.lower()):
            bot_response = f"Your name is {stored_name}."
        elif re.search(r"what(?:'s|s| is) your name\??", user_input.lower()):
            bot_response = "Hi! My name is " + BOT_NAME + "!"
        else:
            prompt = (f"Conversation so far:\n{conversation_history}\n"
                      f"User's question: '{user_input}'\n"
                      "Answer the question directly and concisely.")
            bot_response = generate_response(prompt)

        print("\nBot:", bot_response)
        memory_module.save_memory()


if __name__ == "__main__":
    main()
