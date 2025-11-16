"""
Gradio application for generating multi-turn conversations using distilabel and Claude Agent SDK.
"""

import os
import json
from typing import List, Dict
import gradio as gr
from dotenv import load_dotenv
from distilabel.llms import AnthropicLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from anthropic import Anthropic
import anthropic

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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
            input_mappings={"instruction": "topic"},
            output_mappings={"generation": "question"},
            system_prompt="You are an AI that generates diverse, interesting user questions about a given topic.",
        )

        generate_questions.set_instructions(
            lambda x: f"Generate a unique, specific question that a curious user might ask about {x['topic']}. Make it different from typical questions - be creative and think of various angles (technical, practical, historical, comparative, etc.)."
        )

        load_data >> generate_questions

    # Run the pipeline
    distiset = pipeline.run(use_cache=False)

    questions = []
    for item in distiset["default"]["train"]:
        questions.append({
            "topic": topic,
            "initial_question": item["generation"]
        })

    return questions


def generate_multiturn_conversation_with_claude_agent(topic: str, initial_question: str, num_turns: int = 3) -> List[Dict]:
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
    current_question = initial_question

    for turn in range(num_turns):
        # Add user message
        conversation.append({
            "role": "user",
            "content": current_question
        })

        # Use Claude with web search to generate response
        try:
            # Create a message with web search enabled
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                tools=[
                    {
                        "type": "web_search_20250110",
                        "name": "web_search",
                        "use_web_search": True,
                        "max_results": 5
                    }
                ],
                messages=[
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    } for msg in conversation
                ]
            )

            # Extract the assistant's response
            assistant_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    assistant_content += block.text

            conversation.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Generate a follow-up question for next turn (if not the last turn)
            if turn < num_turns - 1:
                followup_response = client.messages.create(
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
            # If web search fails, fall back to regular Claude
            print(f"Web search error: {e}. Falling back to regular Claude.")
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                messages=[
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    } for msg in conversation
                ]
            )

            assistant_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    assistant_content += block.text

            conversation.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Generate follow-up question
            if turn < num_turns - 1:
                followup_response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Based on this conversation about {topic}, generate a natural follow-up question. Just output the question.\n\nLast question: {current_question}\nLast answer: {assistant_content[:500]}..."
                        }
                    ]
                )
                current_question = followup_response.content[0].text.strip()

    return conversation


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
