import argparse
import logging
from langchain_chroma import Chroma  # Updated import from langchain-chroma package
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer,AutoModelForCausalLM


# Initialize logging to print to console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def load_gpt_neo():
    """Load GPT-Neo model and tokenizer when needed."""
    logging.info("Loading GPT-Neo model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    return model, tokenizer

def load_distilgpt2():
    logging.info("Loading DistilGPT2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    return model, tokenizer

def load_mistral():
    logging.info("Loading Mistral mode and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    return model, tokenizer

def load_minilm():
    logging.info("Loading MiniLM model and tokenizer...")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return model, tokenizer

def query_rag(query_text: str, model_choice: str) -> dict:
    """Handle queries using either RAG pipeline or GPT-Neo."""
    try:
        if model_choice == "mock":
            logging.info("Using mock response...")
            return {"content": f"\nSemantic robustness refers to the ability of a model to maintain accurate predictions and understanding of text even in the presence of lexical and stylistic variations, or meaning-preserving perturbations. It involves ensuring that the model can correctly interpret and process text despite changes in language usage, grammar errors, dialects, or other variations in the input data.\n\nSources:\n1. data/Measure and Improve Robustness in NLP Models A Survey.pdf:4:3\n2. data/Measure and Improve Robustness in NLP Models A Survey.pdf:0:3\n3. data/Measure and Improve Robustness in NLP Models A Survey.pdf:1:6\n4. data/Measure and Improve Robustness in NLP Models A Survey.pdf:0:1\n5. data/Certifiably Robust RAG against Retrieval Corruption.pdf:1:2\n\n", "error": None}



        elif model_choice == "mistral":
            logging.info("Performing Mistral RAG pipeline...")
            # Initialize embedding function and database
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            logging.info("Performing similarity search...")
            results = db.similarity_search_with_score(query_text, k=5)
            logging.debug(f"Search results: {results}")

            # Extract context from search results
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            logging.info("Context extracted for prompt")

            # Prepare the prompt using context and the original query
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            # Load Mistral model
            model, tokenizer = load_mistral()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(inputs["input_ids"], max_length=512, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Process metadata
            sources = [doc.metadata.get("id", None) for doc, _score in results]
            prettified_response = prettify_response(response, sources)
            return {"content": prettified_response, "error": None}

        elif model_choice=='distilgpt2':
            logging.info("Generating response using DistilGPT2...")
            model, tokenizer = load_distilgpt2()
            inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=50)
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"content": response, "error": None}

        elif model_choice=='minilm':
             model, tokenizer = load_minilm()
             logging.info("Generating response using MiniLM...")
             model, tokenizer = load_minilm()
             inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512)
             outputs = model(**inputs)
             response = "Response generated with MiniLM (test simulation)"
             return {"content": response, "error": None}

        elif model_choice == "gpt-neo":
            model, tokenizer = load_gpt_neo()
            logging.info("Generating response using GPT-Neo...")
            inputs = tokenizer.encode(query_text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"content": response, "error": None}

        elif model_choice == "rag":
            logging.info("Initializing RAG pipeline...")
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

            logging.info("Performing similarity search")
            results = db.similarity_search_with_score(query_text, k=5)
            logging.debug(f"Search results: {results}")

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            logging.info("Context extracted for prompt")

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            logging.info("Sending query to OpenAI chat model")
            model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            response_text = model.invoke(prompt)

            logging.info("Processing sources metadata")
            sources = [doc.metadata.get("id", None) for doc, _score in results]

            logging.info("Prettifying response")
            prettified_response = prettify_response(response_text, sources)

            logging.info("Query processing complete")
            return {"content": prettified_response, "error": None}

        else:
            logging.error("Invalid model choice")
            return {"content": None, "error": "Invalid model choice"}

    except Exception as e:
        logging.error(f"Error in query_rag: {e}")
        return {"content": None, "error": str(e)}

def prettify_response(raw_response, sources: list) -> str:
    """Prettify the raw response with source information."""
    logging.debug("Prettifying response")
    try:
        response_content = raw_response.content  # Extract response content
        formatted_sources = "\n".join(
            ["{index}. {source}".format(index=i + 1, source=src.replace('\\', '/')) for i, src in enumerate(sources)]
        )
        prettified_response = f"""
{response_content}

Sources:
{formatted_sources}

"""
        return prettified_response
    except Exception as e:
        logging.error(f"Error in prettifying response: {e}")
        return "Error formatting the response."

def main():
    logging.info("Starting query_data.py")
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, choices=["mock", "gpt-neo", "rag", "minilm","distilgpt2"], default="distilgpt2", help="Model to use for query processing")
    args = parser.parse_args()

    query_text = args.query_text
    model_choice = args.model

    logging.info(f"Received query: {query_text}")
    logging.info(f"Selected model: {model_choice}")
    response = query_rag(query_text, model_choice)
    print(response)

if __name__ == "__main__":
    main()
