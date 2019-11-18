"""Chatbot preprocessosr."""
import chatbot_tools as bot


lines = bot.import_dataset('movie_lines.txt')
conversations = bot.import_dataset('movie_conversations.txt')
id2lines = bot.map_lines(lines)
conversations_ids = bot.get_conversations(conversations)

questions, answers = bot.separate_questions_from_answers(
    conversations_ids
)

mapped_questions = [id2lines[question] for question in questions]
mapped_answers = [id2lines[answer] for answer in answers]

clean_questions = list()
clean_answers = list()

for q_text, a_text in zip(mapped_questions, mapped_answers):
    clean_questions.append(bot.clean_text(q_text))
    clean_answers.append(bot.clean_text(a_text))
