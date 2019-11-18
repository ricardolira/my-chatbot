"""Tests suite for the chatbot."""
import unittest
import chatbot_tools as bot


class ChatbotTests(unittest.TestCase):
    """Tests for chatbot case."""

    def test_if_importing_yields_correct_legth_data(self):
        """Test if data imported is parsed correctly."""
        lines = bot.import_dataset('movie_lines.txt')
        conversations = bot.import_dataset('movie_conversations.txt')

        self.assertEqual(len(lines), 304714)
        self.assertEqual(len(conversations), 83098)

    def test_map_lines_return_ke_and_values_only(self):
        """Test if return from `map_lines` yields a dict {Lxxx: text}."""
        lines = bot.import_dataset('movie_lines.txt')
        mapped_lines = bot.map_lines(lines)
        for key, value in mapped_lines.items():
            payload = key + value

            self.assertNotIn('++$++', payload)

    def test_if_onle_ids_of_conversations_are_returned(self):
        """Test if only ids of the dialogs are returned."""
        conversations = bot.import_dataset('movie_conversations.txt')
        payload = bot.get_conversations(conversations)

        self.assertNotIn('++$++', payload)

    def test_if_questions_are_separated_from_answers(self):
        """Test if questions are followed by answer index."""
        all_conversations = bot.import_dataset('movie_conversations.txt')
        all_conversations = bot.get_conversations(all_conversations)
        questions, answers = bot.separate_questions_from_answers(
            all_conversations
        )
        for question, answer in zip(questions, answers):
            question = int(question.replace('L', ''))
            answer = int(answer.replace('L', ''))

            self.assertTrue(answer - question == 1)

if __name__ == "__main__":
    unittest.main()
