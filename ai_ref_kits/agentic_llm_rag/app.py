
import argparse
import io
import logging
import sys
import time
import warnings
from io import StringIO
from pathlib import Path
from typing import Tuple, Callable

import gradio as gr
import nest_asyncio
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import requests
import yaml
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from llama_index.core.callbacks import CallbackManager
# Agent tools
from tools import PaintCalculator, ShoppingCart
from system_prompt import react_system_header_str

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#Filter unnecessary warnings for demonstration
warnings.filterwarnings("ignore")

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

def setup_models(
    llm_model_path: Path,
    embedding_model_path: Path,
    device: str) -> Tuple[OpenVINOLLM, OpenVINOEmbedding]:
    """
    Sets up LLM and embedding models using OpenVINO.
    
    Args:
        llm_model_path: Path to the LLM model
        embedding_model_path: Path to the embedding model
        device: Target device for inference ("CPU", "GPU", etc.)
        
    Returns:
        Tuple of (llm, embedding) models
    """

    # Load LLM model locally    
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=8192,
        max_new_tokens=500,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},        
        device_map=device,
    )

    # Load the embedding model locally
    embedding = OpenVINOEmbedding(model_id_or_path=str(embedding_model_path), device=device)

    return llm, embedding


def setup_tools()-> Tuple[FunctionTool, FunctionTool, FunctionTool, FunctionTool, FunctionTool]:

    """
    Sets up and returns a collection of tools for paint calculations and shopping cart management.
    
    Returns:
        Tuple containing tools for paint cost calculation, paint gallons calculation, 
        adding items to cart, viewing cart, and clearing cart
    """

    paint_cost_calculator = FunctionTool.from_defaults(
        fn=PaintCalculator.calculate_paint_cost,
        name="calculate_paint_cost",
        description="ALWAYS use this tool when calculating paint cost for a specific area in square feet. Required inputs: area (float, square feet), price_per_gallon (float), add_paint_supply_costs (bool)"
    )

    paint_gallons_calculator = FunctionTool.from_defaults(
    fn=PaintCalculator.calculate_paint_gallons_needed,
    name="calculate_paint_gallons",
    description="Calculate how many gallons of paint are needed to cover a specific area. Required input: area (float, square feet). Returns the number of gallons needed, rounded up to ensure full coverage."
)

    add_to_cart_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.add_to_cart,
        name="add_to_cart",
        description="""
        Use this tool WHENEVER a user wants to add any item to their cart or shopping cart.
        
        PARAMETERS:
        - product_name (string): The exact name of the product (e.g., "Premium Latex Paint")
        - quantity (int): The number of units to add, must be a positive integer (e.g., 2)
        - price_per_unit (float): The price per unit in dollars (e.g., 24.99)
        
        RETURNS:
        - A confirmation message and updated cart contents
        
        EXAMPLES:
        To add 3 gallons of paint at $29.99 each: add_to_cart(product_name="Interior Eggshell Paint", quantity=3, price_per_unit=29.99)
        """
    )
    
    get_cart_items_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.get_cart_items,
        name="view_cart",
        description="""
        Use this tool when a user wants to see what's in their shopping cart.
        No parameters are required.
        
        RETURNS:
        - A list of all items currently in the cart with their details
        
        EXAMPLES:
        To view the current cart contents: view_cart()
        """
    )
    
    clear_cart_tool = FunctionTool.from_defaults(
        fn=ShoppingCart.clear_cart,
        name="clear_cart",
        description="""
        Use this tool when a user asks to empty or clear their shopping cart.
        No parameters are required.
        
        RETURNS:
        - A confirmation message that the cart has been cleared
        
        EXAMPLES:
        To empty the shopping cart: clear_cart()
        """
    )
    return paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, paint_gallons_calculator


def load_documents(text_example_en_path: Path) -> VectorStoreIndex:
    """
    Loads documents from the given path
    
    Args:
        text_example_en_path: Path to the document to load
        
    Returns:
        VectorStoreIndex for the loaded documents
    """
    
    if not text_example_en_path.exists():
        text_example_en = "test_painting_llm_rag.pdf"
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())

    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index

def custom_handle_reasoning_failure(callback_manager: CallbackManager, exception: Exception):
    """
    Provides custom error handling for agent reasoning failures.
    
    Args:
        callback_manager: The callback manager instance for event handling
        exception: The exception that was raised during reasoning
    """
    return "Hmm...I didn't quite that. Could you please rephrase your question to be simpler?"


def run_app(agent: ReActAgent, public_interface: bool = False) -> None:
    """
    Launches the application with the specified agent and interface settings.
    
    Args:
        agent: The ReActAgent instance configured with tools
        public_interface: Whether to launch with a public-facing Gradio interface
    """
    class Capturing(list):
        """A context manager that captures stdout output into a list."""
        def __enter__(self):
            """
            Redirects stdout to a StringIO buffer and returns self.
            Called when entering the 'with' block.
            """
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            """
            Stores captured output in this list and restores stdout.
            Called when exiting the 'with' block.
            """
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout        

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, None)]

    def update_cart_display()-> str:
        """
        Generates an HTML representation of the shopping cart contents.
        
        Retrieves current cart items and creates a formatted HTML table
        showing product details, quantities, prices, and totals.
        If the cart is empty, returns a message indicating this.
        
        Returns:
            str: Markdown-formatted HTML table of cart contents
                or message indicating empty cart
        """
        cart_items = ShoppingCart.get_cart_items()
        if not cart_items:
            return "### 🛒 Your Shopping Cart is Empty"
            
        table = "### 🛒 Your Shopping Cart\n\n"
        table += "<table>\n"
        table += "  <thead>\n"
        table += "    <tr>\n"
        table += "      <th>Product</th>\n"
        table += "      <th>Qty</th>\n"
        table += "      <th>Price</th>\n"
        table += "      <th>Total</th>\n"
        table += "    </tr>\n"
        table += "  </thead>\n"
        table += "  <tbody>\n"
            
        for item in cart_items:
            table += "    <tr>\n"
            table += f"      <td>{item['product_name']}</td>\n"
            table += f"      <td>{item['quantity']}</td>\n"
            table += f"      <td>${item['price_per_unit']:.2f}</td>\n"
            table += f"      <td>${item['total_price']:.2f}</td>\n"
            table += "    </tr>\n"
            
        table += "  </tbody>\n"
        table += "</table>\n"
        
        total = sum(item["total_price"] for item in cart_items)
        table += f"\n**Total: ${total:.2f}**"
        return table

    def _generate_response(chat_history: list, log_history: list | None = None)->Tuple[str,str,str]:
        """
        Generate a streaming response from the agent with formatted thought process logs.
        
        This function:
        1. Captures the agent's thought process
        2. Formats the thought process into readable logs
        3. Streams the agent's response token by token
        4. Tracks performance metrics for thought process and response generation
        5. Updates the shopping cart display
        
        Args:
            chat_history: List of conversation messages
            log_history: List to store logs, will be initialized if None
            
        Yields:
            tuple: (chat_history, formatted_log_history, cart_content)
                - chat_history: Updated with agent's response
                - formatted_log_history: String of joined logs
                - cart_content: HTML representation of the shopping cart
        """
        log.info(f"log_history {log_history}")           
        
        if not isinstance(log_history, list):
            log_history = []

        # Capture time for thought process
        start_thought_time = time.time()

        # Capture the thought process output
        with Capturing() as output:
            try:
                response = agent.stream_chat(chat_history[-1][0])
            except ValueError:
                response = agent.stream_chat(chat_history[-1][0])
        formatted_output = []
        for line in output:
            if "Thought:" in line:
                formatted_output.append("\n🤔 **Thought:**\n" + line.split("Thought:", 1)[1])
            elif "Action:" in line:
                formatted_output.append("\n🔧 **Action:**\n" + line.split("Action:", 1)[1])
            elif "Action Input:" in line:
                formatted_output.append("\n📥 **Input:**\n" + line.split("Action Input:", 1)[1])
            elif "Observation:" in line:
                formatted_output.append("\n📋 **Result:**\n" + line.split("Observation:", 1)[1])
            else:
                formatted_output.append(line)
        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time

        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(formatted_output)
        log_history.append("### 🤔 Agent's Thought Process")
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")
        cart_content = update_cart_display() # update shopping cart
        yield chat_history, "\n".join(log_history), cart_content  # Yield after the thought process time is captured

        # Now capture response generation time
        start_response_time = time.time()

        # Gradually yield the response from the agent to the chat
        # Quick fix for agent occasionally repeating the first word of its repsponse
        last_token = "Dummy Token"
        i = 0
        chat_history[-1][1] = ""
        for token in response.response_gen:
            if i == 0:
                last_token = token
            if i == 1 and token.split()[0] == last_token.split()[0]:
                chat_history[-1][1] += token.split()[1] + " "
            else:
                chat_history[-1][1] += token
            yield chat_history, "\n".join(log_history), cart_content  # Ensure log_history is a string
            if i <= 2: i += 1

        end_response_time = time.time()
        response_time = end_response_time - start_response_time

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s)"

        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history), cart_content  # Join logs into a string for display

    def _reset_chat()-> tuple[str, list, str, str]:
        """
        Resets the chat interface and agent state to initial conditions.
        
        This function:
        1. Resets the agent's internal state
        2. Clears all items from the shopping cart
        3. Returns values needed to reset the UI components
        
        Returns:
            tuple: Values to reset UI components
                - Empty string: Clears the message input
                - Empty list: Resets chat history
                - Default log heading: Sets initial log area text
                - Empty cart display: Shows empty shopping cart
        """
        agent.reset()
        ShoppingCart._cart_items = []
        return "", [], "🤔 Agent's Thought Process", update_cart_display()

    def run()-> None:
        """
        Sets up and launches the Gradio web interface for the Smart Retail Assistant.
        
        This function:
        1. Loads custom CSS styling if available
        2. Configures the Gradio theme and UI components
        3. Sets up the chat interface with agent interaction
        4. Configures event handlers for user inputs
        5. Adds example prompts for users
        6. Launches the web interface
        
        The interface includes:
        - Chat window for user-agent conversation
        - Log window to display agent's thought process
        - Shopping cart display
        - Text input for user messages
        - Submit and Clear buttons
        - Sample questions for easy access
        """
        custom_css = ""
        try:
            with open("css/gradio.css", "r") as css_file:
                custom_css = css_file.read()            
        except Exception as e:            
            log.warning(f"Could not load CSS file: {e}")

        theme = gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "sans-serif"],
        )

        with gr.Blocks(theme=theme, css=custom_css) as demo:

            header = gr.HTML(
                        "<div class='intel-header-wrapper'>"
                        "  <div class='intel-header'>"
                        "    <img src='https://www.intel.com/content/dam/logos/intel-header-logo.svg' class='intel-logo'></img>"
                        "    <div class='intel-title'>Smart Retail Assistant 🤖: Agentic LLMs with RAG 💭</div>"
                        "  </div>"
                        "</div>"
            )

            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Paint Purchase Helper",
                    avatar_images=(None, "https://docs.openvino.ai/2024/_static/favicon.ico"),
                    height=400,  # Adjust height as per your preference
                    scale=2  # Set a higher scale value for Chatbot to make it wider
                    #autoscroll=True,  # Enable auto-scrolling for better UX
                )            
                log_window = gr.Markdown(                                                                    
                        show_label=True,                        
                        value="### 🤔 Agent's Thought Process",
                        height=400,                        
                        elem_id="agent-steps"
                )
                cart_display = gr.Markdown(
                    value=update_cart_display(),
                    elem_id="shopping-cart",
                    height=400
                )

            with gr.Row():
                message = gr.Textbox(label="Ask the Paint Expert 🎨", scale=4, placeholder="Type your prompt/Question and press Enter")

                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear = gr.ClearButton()
                          
            sample_questions = [
                "what paint is the best for kitchens?",
                "what is the price of it?",
                "how many gallons of paint do I need to cover 600 sq ft ?",
                "add them to my cart",
                "what else do I need to complete my project?",
                "add 2 brushes to my cart",
                "create a table with paint products sorted by price",
                "Show me what's in my cart",
                "clear shopping cart",
                "I have a room 1000 sqft, I'm looking for supplies to paint the room"              
            ]
            gr.Examples(
                examples=sample_questions,
                inputs=message, 
                label="Examples"
            )                     
            
            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False                
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, cart_display],
            )

            submit_btn.click(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False,
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, cart_display],
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window, cart_display])

            gr.Markdown("------------------------------")            

        log.info("Demo is ready!")
        demo.queue().launch(share=public_interface)

    run()


def run(chat_model: Path, embedding_model: Path, rag_pdf: Path, device: str, public_interface: bool = False):
    """
    Initializes and runs the agentic rag solution
    
    Args:
        chat_model: Path to the LLM chat model
        embedding_model: Path to the embedding model
        rag_pdf: Path to the PDF file for RAG functionality
        device: Target device for model inference ("CPU", "GPU", "GPU.1")
        public_interface: Whether to expose a public-facing interface
    """
    # Load models and embedding based on parsed arguments
    llm, embedding = setup_models(chat_model, embedding_model, device)

    Settings.embed_model = embedding
    Settings.llm = llm

    # Set up tools
    paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, paint_gallons_calculator = setup_tools()
    
    text_example_en_path = Path(rag_pdf)
    index = load_documents(text_example_en_path)
    log.info(f"loading in {index}")
 
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="""            
            Use this tool for ANY question about paint products, recommendations, prices, or technical specifications.
            
            WHEN TO USE:
            - User asks about paint types, brands, or products
            - User needs price information before adding to cart
            - User needs recommendations based on their project
            - User has technical questions about painting
            
            EXAMPLES:
            - "What paint is best for kitchen cabinets?"
            - "How much does AwesomePainter Interior Acrylic Latex cost?"
            - "What supplies do I need for painting my living room?"
            """,
        ),
    )
    
    nest_asyncio.apply()
 
    # Define agent and available tools
    agent = ReActAgent.from_tools(
        [paint_cost_calculator, add_to_cart_tool, get_cart_items_tool, clear_cart_tool, vector_tool, paint_gallons_calculator],
        llm=llm,
        max_iterations=5,  # Set a max_iterations value
        handle_reasoning_failure_fn=custom_handle_reasoning_failure,
        verbose=True,
        react_chat_formatter=ReActChatFormatter.from_defaults(
            observation_role=MessageRole.TOOL   
        ),
    ) 
    react_system_prompt = PromptTemplate(react_system_header_str)
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})  
    agent.reset()                     
    run_app(agent, public_interface)

if __name__ == "__main__":
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/qwen2-7B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-large-FP32", help="Path to the embedding model directory")
    parser.add_argument("--rag_pdf", type=str, default="data/test_painting_llm_rag.pdf", help="Path to a RAG PDF file with additional knowledge the chatbot can rely on.")    
    parser.add_argument("--device", type=str, default="AUTO:GPU,CPU", help="Device for inferencing (CPU,GPU,GPU.1,NPU)")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()

    run(Path(args.chat_model), Path(args.embedding_model), Path(args.rag_pdf), args.device, args.public)
