"""Chatbot preprocessosr."""
import chatbot_tools as bot


lines = bot.import_dataset('movie_lines.txt')
conversations = bot.import_dataset('movie_conversations.txt')
id2lines = bot.map_lines(lines)
conversations_ids = bot.get_conversations(conversations)
