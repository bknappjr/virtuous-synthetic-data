# Virtuous Synthetic Data - Multi-turn Conversation Generator

A Gradio application that generates high-quality, multi-turn conversations on any topic using Distilabel and Claude Agent SDK with web search capabilities.

## Features

- **Topic-based Generation**: Specify any topic and generate diverse conversations around it
- **Distilabel Integration**: Uses Distilabel to generate varied and unique initial questions
- **Claude Agent SDK**: Leverages Claude's latest model with web search enabled for accurate, up-to-date responses
- **Multi-turn Conversations**: Each conversation contains multiple back-and-forth exchanges
- **Customizable**: Adjust the number of conversations and turns per conversation
- **User-friendly UI**: Clean Gradio interface for easy interaction

## How It Works

1. **Question Generation**: Uses Distilabel to generate diverse initial questions about the specified topic
2. **Conversation Creation**: For each question, creates a multi-turn conversation where:
   - User asks a question
   - Claude responds using web search to provide accurate, current information
   - A follow-up question is generated naturally based on the conversation context
   - This repeats for the specified number of turns
3. **Display**: All conversations are formatted and displayed in an easy-to-read format

## Installation

### Prerequisites

- Python 3.8 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd virtuous-synthetic-data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

## Usage

### Running the Application

Start the Gradio app:
```bash
python app.py
```

The application will launch and be accessible at `http://localhost:7860`

### Using the Interface

1. Enter a topic in the text box (e.g., "Climate Change", "Quantum Computing", "Ancient Rome")
2. Adjust the number of conversations to generate (default: 20)
3. Adjust the number of turns per conversation (default: 3)
4. Click "Generate Conversations"
5. Wait for the conversations to be generated (this may take a few minutes)
6. View the formatted conversations in the output area

## Example Output

```
### Conversation 1

**User:** What are the main causes of climate change?

**Assistant:** Climate change is primarily caused by... [response with web search data]

**User:** How does this compare to historical climate patterns?

**Assistant:** Historical climate data shows... [detailed response]

**User:** What can individuals do to help mitigate these effects?

**Assistant:** There are several effective actions... [actionable advice]

---

### Conversation 2
...
```

## Technical Details

- **Distilabel**: Used for generating diverse initial questions through its pipeline architecture
- **Claude Agent SDK**: Powers the conversation responses with web search integration
- **Model**: Uses Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **Web Search**: Enabled via Anthropic's web search tool for current, accurate information

## Configuration

You can modify the following parameters in the UI:
- Number of conversations (1-50, default: 20)
- Turns per conversation (2-5, default: 3)

## Troubleshooting

### Common Issues

**API Key Error**
- Make sure your `.env` file exists and contains a valid `ANTHROPIC_API_KEY`
- Verify your API key has access to Claude and web search features

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `pip install --upgrade pip`

**Web Search Errors**
- The application will fall back to regular Claude responses if web search is unavailable
- Check the console for specific error messages

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license here]

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for Claude and the Agent SDK
- [Distilabel](https://github.com/argilla-io/distilabel) for the conversation generation framework
- [Gradio](https://gradio.app/) for the web interface
