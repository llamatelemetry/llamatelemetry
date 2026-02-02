"""
llamatelemetry.jupyter - JupyterLab-Specific Features

This module provides JupyterLab-optimized features including:
- Real-time streaming with IPython display
- Progress bars and visual indicators
- Interactive widgets for model controls
- Rich markdown and code rendering
- Chat interfaces with history visualization

Examples:
    Basic streaming in Jupyter:
    >>> from llamatelemetry.jupyter import stream_generate
    >>> stream_generate(engine, "Explain quantum computing")

    Interactive chat widget:
    >>> from llamatelemetry.jupyter import ChatWidget
    >>> chat = ChatWidget(engine)
    >>> chat.display()
"""

from typing import Optional, List, Dict, Any, Callable
import time
import json

# Check if we're in a Jupyter environment
try:
    from IPython.display import display, Markdown, HTML, clear_output, Code
    from IPython import get_ipython
    JUPYTER_AVAILABLE = get_ipython() is not None
except ImportError:
    JUPYTER_AVAILABLE = False
    display = None
    Markdown = None
    HTML = None
    clear_output = None
    Code = None

# Optional dependencies
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None


def is_jupyter_available() -> bool:
    """Check if running in Jupyter environment."""
    return JUPYTER_AVAILABLE


def check_dependencies(require_widgets: bool = False) -> bool:
    """
    Check if required dependencies are available.

    Args:
        require_widgets: Whether to require ipywidgets

    Returns:
        True if dependencies available, False otherwise
    """
    if not JUPYTER_AVAILABLE:
        print("Warning: Not running in Jupyter environment")
        return False

    if require_widgets and not WIDGETS_AVAILABLE:
        print("Warning: ipywidgets not installed. Install with: pip install ipywidgets")
        return False

    return True


def stream_generate(
    engine,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    show_timing: bool = True,
    markdown: bool = True,
    **kwargs
) -> str:
    """
    Stream text generation with real-time display in Jupyter.

    This function uses IPython display to show tokens as they're generated,
    providing immediate visual feedback in Jupyter notebooks.

    Args:
        engine: InferenceEngine instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        show_timing: Display timing information
        markdown: Render output as markdown
        **kwargs: Additional inference parameters

    Returns:
        Complete generated text

    Examples:
        >>> from llamatelemetry import InferenceEngine
        >>> from llamatelemetry.jupyter import stream_generate
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True)
        >>> text = stream_generate(engine, "Write a haiku about AI")
    """
    if not check_dependencies():
        # Fallback to non-streaming
        result = engine.infer(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        print(result.text)
        return result.text

    import requests

    start_time = time.time()

    # Build request payload
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": True,
        **kwargs
    }

    full_text = ""
    token_count = 0

    try:
        # Create display handle for updating
        display_handle = display(Markdown("") if markdown else HTML(""), display_id=True)

        # Stream from server
        response = requests.post(
            f"{engine.server_url}/completion",
            json=payload,
            stream=True,
            timeout=300
        )

        if response.status_code != 200:
            display_handle.update(HTML(f"<span style='color:red'>Error: {response.status_code}</span>"))
            return ""

        # Process streaming chunks
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])

                        # Extract content from chunk
                        if 'content' in data:
                            chunk = data['content']
                            full_text += chunk
                            token_count += 1

                            # Update display
                            if markdown:
                                display_handle.update(Markdown(full_text))
                            else:
                                display_handle.update(HTML(f"<pre>{full_text}</pre>"))

                        # Check if generation is complete
                        if data.get('stop', False):
                            break

                    except json.JSONDecodeError:
                        pass

        # Show timing information
        if show_timing:
            elapsed = time.time() - start_time
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

            timing_html = f"""
            <div style='margin-top: 10px; padding: 8px; background: #f0f0f0; border-radius: 4px; font-size: 0.9em;'>
                <b>Performance:</b> {token_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)
            </div>
            """
            display(HTML(timing_html))

        return full_text

    except Exception as e:
        if JUPYTER_AVAILABLE:
            display(HTML(f"<span style='color:red'>Error: {str(e)}</span>"))
        else:
            print(f"Error: {e}")
        return ""


def progress_generate(
    engine,
    prompts: List[str],
    max_tokens: int = 128,
    **kwargs
) -> List[str]:
    """
    Batch generation with progress bar.

    Args:
        engine: InferenceEngine instance
        prompts: List of prompts to process
        max_tokens: Maximum tokens per prompt
        **kwargs: Additional inference parameters

    Returns:
        List of generated texts
    """
    results = []

    if TQDM_AVAILABLE:
        pbar = tqdm(total=len(prompts), desc="Generating")
        for prompt in prompts:
            result = engine.infer(prompt, max_tokens=max_tokens, **kwargs)
            results.append(result.text if result.success else "")
            pbar.update(1)
        pbar.close()
    else:
        print(f"Processing {len(prompts)} prompts...")
        for i, prompt in enumerate(prompts):
            result = engine.infer(prompt, max_tokens=max_tokens, **kwargs)
            results.append(result.text if result.success else "")
            print(f"  {i+1}/{len(prompts)} complete")

    return results


def display_metrics(engine, as_dataframe: bool = True):
    """
    Display performance metrics in a nice format.

    Args:
        engine: InferenceEngine instance
        as_dataframe: Display as pandas DataFrame if available
    """
    metrics = engine.get_metrics()

    if not JUPYTER_AVAILABLE:
        print(json.dumps(metrics, indent=2))
        return

    # Try pandas first
    try:
        import pandas as pd

        # Flatten metrics
        data = []
        for category, values in metrics.items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    data.append({
                        'Category': category.title(),
                        'Metric': key.replace('_', ' ').title(),
                        'Value': f"{value:.2f}" if isinstance(value, float) else str(value)
                    })

        df = pd.DataFrame(data)
        display(df)
        return
    except ImportError:
        pass

    # Fallback to HTML table
    html = "<table style='border-collapse: collapse; width: 100%;'>"
    html += "<tr style='background: #4CAF50; color: white;'><th style='padding: 8px; text-align: left;'>Category</th><th style='padding: 8px; text-align: left;'>Metric</th><th style='padding: 8px; text-align: right;'>Value</th></tr>"

    row_color = ['#f2f2f2', 'white']
    row_idx = 0

    for category, values in metrics.items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                bg = row_color[row_idx % 2]
                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                html += f"<tr style='background: {bg};'><td style='padding: 8px;'>{category.title()}</td><td style='padding: 8px;'>{key.replace('_', ' ').title()}</td><td style='padding: 8px; text-align: right;'>{value_str}</td></tr>"
                row_idx += 1

    html += "</table>"
    display(HTML(html))


class ChatWidget:
    """
    Interactive chat widget for JupyterLab.

    Provides a rich chat interface with:
    - Text input area
    - Send button
    - Conversation history display
    - Model parameter controls
    - Clear/reset functionality

    Examples:
        >>> from llamatelemetry import InferenceEngine
        >>> from llamatelemetry.jupyter import ChatWidget
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.gguf", auto_start=True)
        >>> chat = ChatWidget(engine)
        >>> chat.display()
    """

    def __init__(
        self,
        engine,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Initialize chat widget.

        Args:
            engine: InferenceEngine instance
            system_prompt: Optional system prompt
            max_tokens: Default max tokens
            temperature: Default temperature
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")

        self.engine = engine
        self.system_prompt = system_prompt
        self.conversation_history = []

        # Create widgets
        self.output = widgets.Output()
        self.input_text = widgets.Textarea(
            placeholder='Type your message here...',
            layout=widgets.Layout(width='100%', height='80px')
        )
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            icon='paper-plane'
        )
        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            icon='trash'
        )
        self.max_tokens_slider = widgets.IntSlider(
            value=max_tokens,
            min=50,
            max=1024,
            step=50,
            description='Max Tokens:',
            style={'description_width': 'initial'}
        )
        self.temperature_slider = widgets.FloatSlider(
            value=temperature,
            min=0.0,
            max=2.0,
            step=0.1,
            description='Temperature:',
            style={'description_width': 'initial'}
        )

        # Event handlers
        self.send_button.on_click(self._on_send)
        self.clear_button.on_click(self._on_clear)

        # Layout
        controls = widgets.HBox([self.send_button, self.clear_button])
        params = widgets.VBox([self.max_tokens_slider, self.temperature_slider])

        self.widget = widgets.VBox([
            widgets.HTML("<h3>Chat Interface</h3>"),
            self.output,
            self.input_text,
            controls,
            params
        ])

    def _on_send(self, button):
        """Handle send button click."""
        user_input = self.input_text.value.strip()
        if not user_input:
            return

        # Clear input
        self.input_text.value = ""

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Display user message
        with self.output:
            display(HTML(f"<div style='background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 8px;'><b>You:</b> {user_input}</div>"))

        # Build prompt with history
        prompt = self._build_prompt()

        # Generate response
        with self.output:
            display(HTML("<div style='background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 8px;'><b>Assistant:</b> <i>Generating...</i></div>"))

        result = self.engine.infer(
            prompt,
            max_tokens=self.max_tokens_slider.value,
            temperature=self.temperature_slider.value
        )

        # Update last message with result
        with self.output:
            clear_output(wait=True)

            # Redisplay all messages
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    display(HTML(f"<div style='background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 8px;'><b>You:</b> {msg['content']}</div>"))

            # Display assistant response
            if result.success:
                self.conversation_history.append({"role": "assistant", "content": result.text})
                display(HTML(f"<div style='background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 8px;'><b>Assistant:</b> {result.text}</div>"))
                display(HTML(f"<div style='font-size: 0.8em; color: #666; margin: 5px 0;'>({result.tokens_generated} tokens, {result.tokens_per_sec:.1f} tok/s)</div>"))
            else:
                display(HTML(f"<div style='background: #ffebee; padding: 10px; margin: 5px 0; border-radius: 8px; color: #c62828;'><b>Error:</b> {result.error_message}</div>"))

    def _on_clear(self, button):
        """Handle clear button click."""
        self.conversation_history = []
        with self.output:
            clear_output()
            display(HTML("<div style='color: #666; font-style: italic;'>Conversation cleared</div>"))

    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        prompt_parts = []

        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}\n")

        for msg in self.conversation_history:
            role = msg["role"].title()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}\n")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def display(self):
        """Display the chat widget."""
        display(self.widget)

        # Show welcome message
        with self.output:
            display(HTML("<div style='color: #666; font-style: italic;'>Chat interface ready. Type a message and click Send.</div>"))


def compare_temperatures(
    engine,
    prompt: str,
    temperatures: List[float] = [0.3, 0.7, 1.0, 1.5],
    max_tokens: int = 100
) -> Dict[float, str]:
    """
    Compare outputs at different temperature settings.

    Useful for exploring model behavior and sampling strategies.

    Args:
        engine: InferenceEngine instance
        prompt: Input prompt
        temperatures: List of temperatures to test
        max_tokens: Maximum tokens per generation

    Returns:
        Dictionary mapping temperature to generated text
    """
    results = {}

    if JUPYTER_AVAILABLE:
        display(HTML(f"<h4>Temperature Comparison</h4><p><b>Prompt:</b> {prompt}</p>"))
    else:
        print(f"Temperature Comparison\nPrompt: {prompt}\n")

    for temp in temperatures:
        result = engine.infer(prompt, max_tokens=max_tokens, temperature=temp)
        results[temp] = result.text if result.success else f"Error: {result.error_message}"

        if JUPYTER_AVAILABLE:
            display(HTML(f"""
            <div style='margin: 15px 0; padding: 12px; border-left: 4px solid #2196F3; background: #f5f5f5;'>
                <div style='font-weight: bold; color: #2196F3; margin-bottom: 8px;'>
                    Temperature: {temp} ({result.tokens_per_sec:.1f} tok/s)
                </div>
                <div style='white-space: pre-wrap;'>{result.text}</div>
            </div>
            """))
        else:
            print(f"\nTemperature: {temp}")
            print(f"{result.text}\n")
            print("-" * 60)

    return results


def visualize_tokens(text: str, engine=None):
    """
    Visualize token boundaries in text (if tokenizer available).

    Args:
        text: Text to visualize
        engine: Optional InferenceEngine with tokenize endpoint
    """
    if not JUPYTER_AVAILABLE:
        print(text)
        return

    # Try to tokenize if engine provided
    if engine:
        try:
            import requests
            response = requests.post(
                f"{engine.server_url}/tokenize",
                json={"content": text}
            )
            if response.status_code == 200:
                tokens = response.json().get('tokens', [])

                # Create HTML with token boundaries
                html = "<div style='font-family: monospace; line-height: 1.8;'>"
                for token in tokens:
                    html += f"<span style='border: 1px solid #ddd; padding: 2px 4px; margin: 2px; display: inline-block; background: #f9f9f9;'>{token}</span>"
                html += "</div>"

                display(HTML(html))
                display(HTML(f"<p><b>Total tokens:</b> {len(tokens)}</p>"))
                return
        except Exception as e:
            pass

    # Fallback to simple display
    display(Markdown(text))
