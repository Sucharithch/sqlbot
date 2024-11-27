import streamlit as st
import time

class StreamlitUICallbackHandler:
    def __init__(self, model):
        self.model = model
        self.final_message = ""
        self.loading_message = None

    def start_loading_message(self):
        """Start the loading animation."""
        self.loading_message = st.empty()
        self.loading_message.markdown("🤔 Thinking...")

    def update_loading_message(self, message):
        """Update the loading message."""
        if self.loading_message:
            self.loading_message.markdown(f"🤔 {message}")

    def end_loading_message(self):
        """End the loading animation."""
        if self.loading_message:
            self.loading_message.empty()

def message_func(message, is_user=False, is_df=False, model=""):
    """Display a message in the Streamlit chat interface."""
    if is_user:
        avatar = "👤"  # User avatar emoji
        align = "right"
        bg_color = "bg-blue-100"
    else:
        avatar = "🤖"  # Bot avatar emoji
        align = "left"
        bg_color = "bg-gray-100"

    st.markdown(
        f"""
        <div class="chat-row {align}">
            <div class="chat-bubble {bg_color}">
                <div class="chat-avatar">{avatar}</div>
                <div class="chat-message">
                    {message if not is_df else '<pre>' + message + '</pre>'}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add any custom CSS needed for chat styling
st.markdown("""
<style>
.chat-row {
    display: flex;
    margin-bottom: 1rem;
}
.chat-row.right {
    justify-content: flex-end;
}
.chat-bubble {
    padding: 1rem;
    border-radius: 1rem;
    max-width: 80%;
}
.chat-avatar {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.chat-message pre {
    white-space: pre-wrap;
    word-wrap: break-word;
}
.bg-blue-100 {
    background-color: #dbeafe;
}
.bg-gray-100 {
    background-color: #f3f4f6;
}
</style>
""", unsafe_allow_html=True)
