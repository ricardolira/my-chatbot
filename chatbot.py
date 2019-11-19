"""Chatbot preprocessosr."""
import chatbot_tools as bot

# importing dataset
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


word2count = dict()

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

threshold = 20

questionswords2int = dict()
word_number = 0

for word, count in word2count.items():
    if count > threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerwords2int = dict()
word_number = 0

for word, count in word2count.items():
    if count > threshold:
        answerwords2int[word] = word_number
        word_number += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    answerwords2int[token] = len(answerwords2int) + 1

answerint2words = {w_i: w for w, w_i in answerwords2int.items()}

clean_answers = [answer + ' <EOS>' for answer in clean_answers]

questions_to_int = list()
for question in clean_questions:
    ints = list()
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
        questions_to_int.append(ints)

answers_to_int = list()
for question in clean_questions:
    ints = list()
    for word in question.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
        answers_to_int.append(ints)

# sorting questions and answers into length oriented to optimize training
sorted_clean_questions = list()
sorted_clean_answers = list()
max_words = 25
for length in range(1, max_words + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])

inputs, targets, lr, keep_prob = bot.model_inputs()
