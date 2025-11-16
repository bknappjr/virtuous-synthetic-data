"""
Gradio application for generating multi-turn conversations using distilabel and Claude Agent SDK.
"""

import os
import json
import asyncio
import signal
import threading
from typing import List, Dict
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
    with Pipeline(name="question-generation") as pipeline:
        load_data = LoadDataFromDicts(
            name="load_topics",
            data=[{"topic": topic} for _ in range(num_conversations)]
        )

        generate_questions = TextGeneration(
            name="generate_questions",
            llm=AnthropicLLM(
                model="claude-sonnet-4-5-20250929",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            template="Generate a unique, specific question that a curious user might ask about {{ topic }}. Make it different from typical questions - be creative and think of various angles (technical, practical, historical, comparative, etc.).",
            columns=["topic"],
            output_mappings={"generation": "question"},
            system_prompt="You are an AI that generates diverse, interesting user questions about a given topic.",
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


async def generate_multiturn_conversation_async(topic: str, initial_question: str, num_turns: int = 3) -> List[Dict]:
    """
    Generate a multi-turn conversation using Claude Agent SDK with web search.

    Args:
        topic: The main topic of conversation
        initial_question: The initial user question
        num_turns: Number of conversation turns

    Returns:
        List of conversation messages
    """
    conversation = []

    # Configure Claude Agent SDK with WebSearch enabled
    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch"],
        permission_mode='allow'
    )

    # Create a ClaudeSDKClient to maintain conversation context
    sdk_client = ClaudeSDKClient(options=options)

    current_question = initial_question

    for turn in range(num_turns):
        # Add user message to our conversation log
        conversation.append({
            "role": "user",
            "content": current_question
        })

        try:
            # Send the question to Claude Agent SDK
            # The SDK maintains conversation history automatically
            full_response = ""
            async for message in sdk_client.send_message(current_question):
                # Accumulate the response text
                full_response += str(message)

            assistant_content = full_response.strip()

            # Add assistant response to conversation log
            conversation.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Generate a follow-up question for next turn (if not the last turn)
            if turn < num_turns - 1:
                followup_response = anthropic_client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Based on this conversation about {topic}, generate a natural follow-up question that would continue the discussion. Just output the question, nothing else.\n\nLast question: {current_question}\nLast answer: {assistant_content[:500]}..."
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
            break

    return conversation


def generate_multiturn_conversation_with_claude_agent(topic: str, initial_question: str, num_turns: int = 3) -> List[Dict]:
    """
    Wrapper function to run the async conversation generation.

    Args:
        topic: The main topic of conversation
        initial_question: The initial user question
        num_turns: Number of conversation turns

    Returns:
        List of conversation messages
    """
    return asyncio.run(generate_multiturn_conversation_async(topic, initial_question, num_turns))


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


def generate_conversations(topic: str, num_conversations: int = 20, turns_per_conversation: int = 3, progress=gr.Progress()) -> str:
    """
    Main function to generate conversations about a topic.

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

    Enter a topic below and click "Generate Conversations" to create 20 multi-turn conversations.
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

    output = gr.Markdown(label="Generated Conversations")

    generate_btn.click(
        fn=generate_conversations,
        inputs=[topic_input, num_conversations, num_turns],
        outputs=output
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
