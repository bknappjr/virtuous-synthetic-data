"""
Gradio application for generating multi-turn conversations using distilabel and Claude Agent SDK.
"""

import os
import json
import asyncio
import signal
import threading
import zipfile
import tempfile
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional
import gradio as gr
from dotenv import load_dotenv
from distilabel.llms import AnthropicLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Initialize Anthropic client for initial question generation
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Monkey-patch signal.signal to avoid errors when running in non-main threads (e.g., Gradio worker threads)
_original_signal = signal.signal

def _thread_safe_signal(signalnum, handler):
    """
    Wrapper for signal.signal that only works in the main thread.
    In worker threads, it returns a dummy handler to avoid ValueError.
    """
    if threading.current_thread() is threading.main_thread():
        return _original_signal(signalnum, handler)
    else:
        # Return a dummy handler when not in main thread
        # This prevents the ValueError while allowing the code to continue
        return signal.SIG_DFL

# Apply the monkey-patch
signal.signal = _thread_safe_signal


def generate_user_questions(topic: str, num_conversations: int = 20) -> List[Dict]:
    """Generate initial user questions about the topic using distilabel."""

    # Create a pipeline to generate diverse user questions
    # Add an index to each item to encourage unique questions
    with Pipeline(name="question-generation") as pipeline:
        load_data = LoadDataFromDicts(
            name="load_topics",
            data=[{"topic": topic, "question_number": i+1} for i in range(num_conversations)]
        )

        generate_questions = TextGeneration(
            name="generate_questions",
            llm=AnthropicLLM(
                model="claude-3-5-haiku-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            template="Generate unique question #{{ question_number }} about {{ topic }}. Make this question COMPLETELY DIFFERENT from any other question someone might ask. Consider various angles: technical details, practical applications, historical context, ethical implications, future trends, comparisons, beginner vs expert perspectives, real-world examples, edge cases, or controversies. Output ONLY the question, nothing else.",
            columns=["topic", "question_number"],
            output_mappings={"generation": "question"},
            system_prompt="You are an AI that generates highly diverse, unique user questions. Each question must explore a different aspect or angle of the topic. Avoid repetition and generic questions.",
        )

        load_data >> generate_questions

    # Run the pipeline
    distiset = pipeline.run(use_cache=False)

    questions = []
    for item in distiset["default"]["train"]:
        questions.append({
            "topic": topic,
            "initial_question": item["question"]
        })

    return questions


async def generate_multiturn_conversation_async(topic: str, initial_question: str, num_turns: int = 3, update_callback=None) -> List[Dict]:
    """
    Generate a multi-turn conversation using Claude Agent SDK with web search.

    Args:
        topic: The main topic of conversation
        initial_question: The initial user question
        num_turns: Number of conversation turns
        update_callback: Optional callback function to call after each turn with current conversation state

    Returns:
        List of conversation messages
    """
    conversation = []

    # Configure Claude Agent SDK with WebSearch enabled
    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch"],
        permission_mode='bypassPermissions',
        model='claude-3-5-haiku-20241022'
    )

    current_question = initial_question

    try:
        # Use ClaudeSDKClient as async context manager to maintain conversation context
        async with ClaudeSDKClient(options=options) as sdk_client:
            for turn in range(num_turns):
                # Add user message to our conversation log
                conversation.append({
                    "role": "user",
                    "content": current_question
                })

                # Call update callback after user message
                if update_callback:
                    update_callback(conversation)

                try:
                    # Send the question to Claude Agent SDK using the correct API
                    await sdk_client.query(current_question)

                    # Receive the response with real-time streaming
                    full_response = ""

                    # Add a placeholder for the assistant response that we'll update during streaming
                    conversation.append({
                        "role": "assistant",
                        "content": ""
                    })

                    async for message in sdk_client.receive_response():
                        # Accumulate the response text
                        full_response += str(message)

                        # Update the last message (assistant response) with streaming content
                        conversation[-1]["content"] = full_response

                        # Call update callback with streaming updates
                        if update_callback:
                            update_callback(conversation)

                    assistant_content = full_response.strip()

                    # Update with final cleaned content
                    conversation[-1]["content"] = assistant_content

                    # Call update callback after assistant response is complete
                    if update_callback:
                        update_callback(conversation)

                    # Generate a follow-up question for next turn (if not the last turn)
                    if turn < num_turns - 1:
                        followup_response = anthropic_client.messages.create(
                            model="claude-3-5-haiku-20241022",
                            max_tokens=200,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Read the following answer carefully and generate a natural follow-up question that builds directly on the information provided in the answer. The question should dig deeper into a specific point mentioned in the answer, ask for clarification, or explore a related aspect discussed in the response. Output ONLY the question, nothing else.\n\nTopic: {topic}\n\nPrevious question: {current_question}\n\nAnswer to use as basis for follow-up:\n{assistant_content[:1500]}"
                                }
                            ]
                        )

                        current_question = followup_response.content[0].text.strip()

                except Exception as e:
                    # Log error and use a simpler response
                    print(f"Error generating response: {e}")
                    error_msg = f"Error generating response for this turn. Please check your API key and connection."
                    conversation.append({
                        "role": "assistant",
                        "content": error_msg
                    })

                    # Call update callback even on error
                    if update_callback:
                        update_callback(conversation)
                    break

    except Exception as e:
        print(f"Error initializing Claude SDK client: {e}")
        conversation.append({
            "role": "assistant",
            "content": f"Error initializing conversation. Please check your API key and connection. Error: {str(e)}"
        })

        # Call update callback even on error
        if update_callback:
            update_callback(conversation)

    return conversation


def generate_multiturn_conversation_with_claude_agent(topic: str, initial_question: str, num_turns: int = 3, update_callback=None) -> List[Dict]:
    """
    Wrapper function to run the async conversation generation.

    Args:
        topic: The main topic of conversation
        initial_question: The initial user question
        num_turns: Number of conversation turns
        update_callback: Optional callback function to call after each turn

    Returns:
        List of conversation messages
    """
    return asyncio.run(generate_multiturn_conversation_async(topic, initial_question, num_turns, update_callback))


def format_conversations_for_display(conversations: List[List[Dict]]) -> str:
    """Format the generated conversations for display in Gradio."""
    output = ""

    for i, conversation in enumerate(conversations, 1):
        output += f"### Conversation {i}\n\n"

        for msg in conversation:
            role = msg["role"].capitalize()
            content = msg["content"]
            output += f"**{role}:** {content}\n\n"

        output += "---\n\n"

    return output


def format_conversation_as_html_accordion(
    conversation_num: int,
    conversation: List[Dict],
    is_open: bool = False,
    is_streaming: bool = False
) -> str:
    """Format a single conversation as an HTML accordion using details/summary."""
    open_attr = "open" if is_open else ""
    status = "🔄 In Progress..." if is_streaming else "✅ Complete"

    html = f'<details {open_attr} style="margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: white; opacity: 1;">\n'
    html += f'<summary style="cursor: pointer; font-weight: bold; font-size: 16px; padding: 10px; background-color: #e9e9e9; border-radius: 5px; margin: -10px; margin-bottom: 10px; opacity: 1;">'
    html += f'Conversation {conversation_num} {status}'
    html += '</summary>\n'
    html += '<div style="padding: 15px; background-color: white; border-radius: 5px; margin-top: 10px; opacity: 1;">\n'

    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"].replace("\n", "<br>")

        if role == "User":
            html += f'<div style="margin-bottom: 15px; padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 4px; opacity: 1;">\n'
            html += f'<strong style="color: #1976D2; opacity: 1;">👤 User:</strong><br>\n'
            html += f'<span style="color: #333; opacity: 1;">{content}</span>\n'
            html += '</div>\n'
        else:
            html += f'<div style="margin-bottom: 15px; padding: 10px; background-color: #f3e5f5; border-left: 4px solid #9C27B0; border-radius: 4px; opacity: 1;">\n'
            html += f'<strong style="color: #7B1FA2; opacity: 1;">🤖 Assistant:</strong><br>\n'
            html += f'<span style="color: #333; opacity: 1;">{content}</span>\n'
            html += '</div>\n'

    html += '</div>\n'
    html += '</details>\n'

    return html


def format_all_conversations_html(
    all_conversations: List[List[Dict]],
    current_conversation_idx: int = -1,
    current_conversation_content: List[Dict] = None
) -> str:
    """Format all conversations as HTML accordions with streaming support."""
    # Header section
    html = '<div style="border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; margin-bottom: 15px;">\n'
    html += '<h3 style="margin: 0; color: #333;">💬 Generated Conversations</h3>\n'

    total = len(all_conversations)
    if current_conversation_idx >= 0:
        html += f'<p style="margin: 5px 0; color: #666; font-size: 14px;">Completed: {total} | In Progress: Conversation {current_conversation_idx + 1}</p>\n'
    else:
        html += f'<p style="margin: 5px 0; color: #666; font-size: 14px;">Total Conversations: {total}</p>\n'

    html += '</div>\n'

    # Scrollable container for conversations
    html += '<div style="max-height: 600px; overflow-y: auto; padding: 10px; background-color: white; border-radius: 8px; opacity: 1;">\n'

    # Add completed conversations (collapsed)
    for i, conversation in enumerate(all_conversations):
        html += format_conversation_as_html_accordion(i + 1, conversation, is_open=False, is_streaming=False)

    # Add current streaming conversation (expanded)
    if current_conversation_idx >= 0 and current_conversation_content:
        html += format_conversation_as_html_accordion(
            current_conversation_idx + 1,
            current_conversation_content,
            is_open=True,
            is_streaming=True
        )

    if len(all_conversations) == 0 and current_conversation_idx < 0:
        html += '<p style="color: #888; text-align: center; padding: 30px; font-style: italic;">No conversations yet...</p>\n'

    html += '</div>\n'

    return html


def format_conversation_as_markdown(conversation: List[Dict], conversation_num: int, topic: str = "") -> str:
    """
    Format a single conversation as a markdown document.

    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        conversation_num: The conversation number for the title
        topic: Optional topic to include in the header

    Returns:
        Formatted markdown string
    """
    markdown = f"# Conversation {conversation_num}\n\n"

    if topic:
        markdown += f"**Topic:** {topic}\n\n"

    markdown += "---\n\n"

    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"]

        if role == "User":
            markdown += f"## 👤 User\n\n{content}\n\n"
        else:
            markdown += f"## 🤖 Assistant\n\n{content}\n\n"

        markdown += "---\n\n"

    return markdown


def generate_zip_file(all_conversations: List[List[Dict]], topic: str = "") -> Optional[str]:
    """
    Generate a zip file containing markdown documents for each conversation.

    Args:
        all_conversations: List of conversations, where each conversation is a list of messages
        topic: Optional topic string to include in filenames and documents

    Returns:
        Path to the generated zip file, or None if generation fails
    """
    if not all_conversations:
        return None

    try:
        # Create a temporary directory for markdown files
        temp_dir = tempfile.mkdtemp()

        # Generate markdown files for each conversation
        for i, conversation in enumerate(all_conversations, 1):
            markdown_content = format_conversation_as_markdown(conversation, i, topic)

            # Create a safe filename
            filename = f"conversation_{i:03d}.md"
            filepath = Path(temp_dir) / filename

            # Write markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

        # Create zip file
        zip_path = Path(temp_dir) / "conversations.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all markdown files to the zip
            for md_file in Path(temp_dir).glob("*.md"):
                zipf.write(md_file, md_file.name)

        return str(zip_path)

    except Exception as e:
        print(f"Error generating zip file: {e}")
        return None


def generate_conversations_streaming(topic: str, num_conversations: int = 20, turns_per_conversation: int = 3, progress=gr.Progress()):
    """
    Generator function to stream conversation updates in real-time.

    Args:
        topic: The topic to generate conversations about
        num_conversations: Number of conversations to generate (default: 20)
        turns_per_conversation: Number of turns per conversation (default: 3)

    Yields:
        Tuple of (HTML string with accordion view of conversations, File update, download section visibility)
    """
    if not topic or topic.strip() == "":
        yield "<p style='color: red;'>Please enter a valid topic.</p>", gr.update(value=None, visible=False), gr.update(visible=False)
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        yield "<p style='color: red;'>Error: ANTHROPIC_API_KEY environment variable not set. Please create a .env file with your API key.</p>", gr.update(value=None, visible=False), gr.update(visible=False)
        return

    progress(0, desc="Generating initial questions...")
    yield "<p style='color: #888; text-align: center; padding: 20px;'>⏳ Generating initial questions...</p>", gr.update(value=None, visible=False), gr.update(visible=False)

    # Step 1: Generate diverse initial questions using distilabel
    try:
        questions = generate_user_questions(topic, num_conversations)
    except Exception as e:
        yield f"<p style='color: red;'>Error generating questions: {str(e)}</p>", gr.update(value=None, visible=False), gr.update(visible=False)
        return

    yield "<p style='color: #888; text-align: center; padding: 20px;'>✅ Questions generated! Starting conversations...</p>", gr.update(value=None, visible=False), gr.update(visible=False)

    # Step 2: Generate multi-turn conversations for each question
    all_conversations = []

    # Shared state for streaming updates
    streaming_state = {"current_conv": []}

    for i, q in enumerate(questions):
        progress((i + 1) / len(questions), desc=f"Generating conversation {i+1}/{num_conversations}...")

        # Reset streaming state for new conversation
        streaming_state["current_conv"] = []

        # Define callback to update UI in real-time during conversation
        def streaming_update_callback(conversation_snapshot):
            # Store the current state and yield to UI
            streaming_state["current_conv"] = conversation_snapshot.copy()

        # Generate the conversation with streaming updates
        try:
            # Create a wrapper that yields updates during generation
            async def generate_with_yields():
                return await generate_multiturn_conversation_async(
                    topic=q["topic"],
                    initial_question=q["initial_question"],
                    num_turns=turns_per_conversation,
                    update_callback=streaming_update_callback
                )

            # Run async function in a thread-safe way
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Use a custom approach to yield during execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: loop.run_until_complete(generate_with_yields()))

                # Poll for updates while the task is running
                while not future.done():
                    if streaming_state["current_conv"]:
                        # Yield current state while conversation is in progress
                        yield format_all_conversations_html(all_conversations, i, streaming_state["current_conv"]), gr.update(value=None, visible=False), gr.update(visible=False)
                    time.sleep(0.05)  # Small delay to avoid overwhelming the UI

                # Get the final conversation
                conversation = future.result()

            loop.close()

            # Add completed conversation
            all_conversations.append(conversation)

            # Yield with conversation completed and collapsed
            yield format_all_conversations_html(all_conversations, -1, None), gr.update(value=None, visible=False), gr.update(visible=False)

        except Exception as e:
            if 'loop' in locals():
                loop.close()
            error_conv = [{"role": "assistant", "content": f"Error: {str(e)}"}]
            all_conversations.append(error_conv)
            yield format_all_conversations_html(all_conversations, -1, None), gr.update(value=None, visible=False), gr.update(visible=False)

    progress(1.0, desc="Complete!")

    # Generate zip file with all conversations
    zip_path = generate_zip_file(all_conversations, topic)

    # Final yield with all conversations and the zip file (now visible)
    yield format_all_conversations_html(all_conversations, -1, None), gr.update(value=zip_path, visible=True), gr.update(visible=True)


def generate_conversations(topic: str, num_conversations: int = 20, turns_per_conversation: int = 3, progress=gr.Progress()) -> str:
    """
    Main function to generate conversations about a topic (non-streaming version for backward compatibility).

    Args:
        topic: The topic to generate conversations about
        num_conversations: Number of conversations to generate (default: 20)
        turns_per_conversation: Number of turns per conversation (default: 3)

    Returns:
        Formatted string of all conversations
    """
    if not topic or topic.strip() == "":
        return "Please enter a valid topic."

    if not os.getenv("ANTHROPIC_API_KEY"):
        return "Error: ANTHROPIC_API_KEY environment variable not set. Please create a .env file with your API key."

    progress(0, desc="Generating initial questions...")

    # Step 1: Generate diverse initial questions using distilabel
    questions = generate_user_questions(topic, num_conversations)

    # Step 2: Generate multi-turn conversations for each question
    all_conversations = []

    for i, q in enumerate(questions):
        progress((i + 1) / len(questions), desc=f"Generating conversation {i+1}/{num_conversations}...")

        conversation = generate_multiturn_conversation_with_claude_agent(
            topic=q["topic"],
            initial_question=q["initial_question"],
            num_turns=turns_per_conversation
        )

        all_conversations.append(conversation)

    progress(1.0, desc="Complete!")

    # Format and return
    return format_conversations_for_display(all_conversations)


# Create Gradio interface
with gr.Blocks(title="Multi-turn Conversation Generator") as demo:
    gr.Markdown("""
    # Multi-turn Conversation Generator

    This application generates multi-turn conversations about any topic using:
    - **Distilabel** for generating diverse initial questions
    - **Claude Agent SDK** with web search for generating informative responses

    Enter a topic below and click "Generate Conversations" to create multi-turn conversations.
    """)

    with gr.Row():
        with gr.Column():
            topic_input = gr.Textbox(
                label="Topic",
                placeholder="e.g., Climate Change, Quantum Computing, Ancient Rome...",
                lines=1
            )

            with gr.Row():
                num_conversations = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    label="Number of Conversations"
                )

                num_turns = gr.Slider(
                    minimum=2,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Turns per Conversation"
                )

            generate_btn = gr.Button("Generate Conversations", variant="primary")

    gr.Markdown("## 📊 Live Conversation Output")
    gr.Markdown("Watch conversations generate in real-time below. Each conversation expands while active and collapses when complete.")

    # Streaming accordion output view
    accordion_output = gr.HTML(
        label="Streaming Conversations",
        value="<p style='color: #888; text-align: center; padding: 20px;'>Click 'Generate Conversations' to begin...</p>"
    )

    # File download output (hidden until ready)
    with gr.Column(visible=False) as download_section:
        gr.Markdown("## 📦 Download Conversations")
        download_output = gr.File(
            label="Download ZIP File",
            file_types=[".zip"],
            type="filepath",
            visible=True
        )

    # Wire up the streaming function
    generate_btn.click(
        fn=generate_conversations_streaming,
        inputs=[topic_input, num_conversations, num_turns],
        outputs=[accordion_output, download_output, download_section],
        show_progress="hidden"
    )

    gr.Markdown("""
    ### Setup Instructions
    1. Install dependencies: `pip install -r requirements.txt`
    2. Create a `.env` file with your `ANTHROPIC_API_KEY`
    3. Run the app: `python app.py`

    **Note:** This app requires an Anthropic API key with access to Claude and web search capabilities.
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
