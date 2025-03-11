# Cell 1: Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap

# Cell 2: Define the DocumentSummarizer class
class DocumentSummarizer:
    def __init__(self, model_id="meta-llama/Llama-2-70b-hf", device="cuda"):
        """
        Initialize the DocumentSummarizer with the Llama model.
        
        Args:
            model_id (str): The model identifier for Hugging Face.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            load_in_8bit=self.device == "cuda",  # Enable 8-bit quantization for GPU
        )
        
    def create_prompt(self, document_text):
        """
        Creates a detailed prompt for the Llama model to summarize the document.
        
        Args:
            document_text (str): The text to be summarized.
            
        Returns:
            str: The complete prompt for the model.
        """
        prompt = f"""
You are an expert document analyst and summarizer. Your task is to provide a comprehensive summary of the following document.

INSTRUCTIONS:
1. Create a concise summary that captures the main points, key arguments, and essential information.
2. Identify and include only the most important details from the text.
3. Structure your summary in a logical manner, with clear sections if necessary.
4. Use ONLY information directly stated in the provided text.
5. DO NOT add any external information, opinions, or interpretations that are not present in the original document.
6. DO NOT make assumptions or inferences beyond what is explicitly stated.
7. If the document contains data, statistics, or specific facts, include them in your summary with their exact values.
8. For each main point in your summary, briefly indicate where in the document this information comes from (e.g., "As stated in paragraph 2...").
9. If any information is ambiguous or unclear in the original text, acknowledge this in your summary rather than attempting to clarify with external knowledge.
10. Format your response as follows:
   - EXECUTIVE SUMMARY: A 2-3 sentence overview of the entire document
   - KEY POINTS: Bulleted list of the most important information
   - DETAILED SUMMARY: A more comprehensive summary broken into relevant sections
   - SUPPORTING EVIDENCE: Direct quotes or specific references from the text that support the main points

Now, provide a summary of the following document:

{document_text}

Remember, your summary must be factual, accurate, and based ONLY on the information provided in the document.
"""
        return prompt

    def summarize(self, document_text, max_length=4096, temperature=0.1):
        """
        Summarize the provided document text.
        
        Args:
            document_text (str): The text to be summarized.
            max_length (int): Maximum length of the generated summary.
            temperature (float): Controls randomness in generation (lower = more deterministic).
            
        Returns:
            str: The generated summary.
        """
        prompt = self.create_prompt(document_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate summary with specified parameters
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                num_return_sequences=1
            )
        
        summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the summary part (remove the prompt)
        summary = summary[len(prompt):]
        
        return summary.strip()

# Cell 3: Create function to load document from file
def load_document(file_path):
    """
    Load document text from a file.
    
    Args:
        file_path (str): Path to the document file.
        
    Returns:
        str: The document text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Cell 4: Create function to save summary to file
def save_summary(summary, file_path):
    """
    Save the generated summary to a file.
    
    Args:
        summary (str): The generated summary.
        file_path (str): Path to save the summary file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved to {file_path}")

# Cell 5: Initialize the summarizer
# You can change the model ID if you want to use a different version
model_id = "meta-llama/Llama-2-70b-hf"  # Change this if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the summarizer
summarizer = DocumentSummarizer(model_id=model_id, device=device)

# Cell 6: Example usage with a sample document
# Option 1: Load from file
# document_text = load_document("path/to/your/document.txt")

# Option 2: Sample text directly in the notebook for testing
document_text = """
[Insert your document text here for summarization]
"""

# Cell 7: Generate the summary
summary = summarizer.summarize(
    document_text,
    max_length=4096,  # Adjust based on your needs
    temperature=0.1    # Lower for more deterministic outputs
)

# Cell 8: Display the summary
print("\n" + "="*80 + "\n")
print("DOCUMENT SUMMARY:")
print("\n" + "="*80 + "\n")
print(textwrap.fill(summary, width=100))
print("\n" + "="*80 + "\n")

# Cell 9: Save the summary (optional)
# Uncomment the line below to save the summary to a file
# save_summary(summary, "output_summary.txt")
