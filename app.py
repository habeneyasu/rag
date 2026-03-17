import gradio as gr
from dotenv import load_dotenv
from openai import InternalServerError, RateLimitError, APIConnectionError, APIError

from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(context):
    result = "## Relevant Context\n\n"
    for doc in context:
        result += f"### From {doc.metadata['source']}\n\n"
        result += doc.page_content + "\n\n"
    return result


def respond(message, history):
    """Handle user message and generate response."""
    if not message or not message.strip():
        return history, "*Please enter a question*"
    
    # Add user message to history
    if history is None:
        history = []
    history = history + [{"role": "user", "content": message}]
    
    try:
        # Get prior conversation (all messages except the last one we just added)
        prior = history[:-1] if len(history) > 1 else []
        answer, context = answer_question(message, prior)
        history.append({"role": "assistant", "content": answer})
        return history, format_context(context)
    except (InternalServerError, RateLimitError, APIConnectionError, APIError) as e:
        error_message = (
            "⚠️ **Service Temporarily Unavailable**\n\n"
            "The AI service is currently experiencing issues. This is usually temporary.\n\n"
            "**What you can do:**\n"
            "- Please try again in a few moments\n"
            "- The system will automatically retry on the next attempt\n\n"
            f"*Error details: {type(e).__name__}*"
        )
        history.append({"role": "assistant", "content": error_message})
        return history, "*Unable to retrieve context due to service error*"
    except Exception as e:
        error_message = (
            "❌ **An error occurred**\n\n"
            "Something unexpected happened while processing your question.\n\n"
            "**What you can do:**\n"
            "- Please try rephrasing your question\n"
            "- If the problem persists, the service may be temporarily unavailable\n\n"
            f"*Error: {str(e)[:200]}*"
        )
        history.append({"role": "assistant", "content": error_message})
        return history, "*Unable to retrieve context due to error*"


def main():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant", theme=theme) as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", 
                    height=600, 
                    type="messages", 
                    show_copy_button=True,
                    value=[]
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        # Simplified event handler - single function handles everything
        message.submit(
            fn=respond,
            inputs=[message, chatbot],
            outputs=[chatbot, context_markdown]
        ).then(
            lambda: "",  # Clear the message box after submission
            outputs=[message]
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
