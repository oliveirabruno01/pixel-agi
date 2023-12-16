import unittest
from unittest.mock import patch, MagicMock


class TestApp(unittest.TestCase):
    @patch('openai.OpenAI')
    def test_ai_response(self, mock_openai):
        # Mock the OpenAI client and its chat.completions.create method
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='mocked content'))])

        # Import the ai_response function from your module (replace 'your_module' with the actual module name)
        from app import ai_response

        # Define the input parameters for ai_response
        input_text = 'test input'
        temperature = 0.01
        n_shots = 1
        n_shots_size = 800
        task_type = 'general'
        stream = False
        api_input = 'test_api_key'
        base_url_input = 'test_base_url'
        base_model = 'test_base_model'

        # Define the expected messages array
        base_system_prompt = {"role": "system", "content": "Your expected system prompt"}
        in_context_learning = [{"role": "user", "content": "Your expected in-context learning"}]
        context = [
            {"role": "system", "content": 'test finished, the above messages are just examples to guide you on the following tasks. the following is totally unrelated with the above messages, it\'s a brand new interaction. don\'t mix things!'},
            {"role": "user", "name": "real user", "content": input_text}
        ]
        expected_messages = [base_system_prompt, *in_context_learning, *context]

        # Call ai_response with the input parameters
        result = ai_response(input_text, temperature, n_shots, n_shots_size, task_type, stream, api_input,
                             base_url_input, base_model)

        # Assert that the OpenAI client was called with the correct parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model=base_model,
            temperature=temperature,
            messages=expected_messages,
            max_tokens=5000
        )

        # Assert that the result is as expected
        self.assertEqual(result, ('mocked content', '', None, None, None))

if __name__ == '__main__':
    unittest.main()
