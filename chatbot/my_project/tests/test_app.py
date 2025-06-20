import unittest
import os
import io # For BytesIO
from chatbot.my_project.app import app, allowed_file
import chatbot.my_project.app as app_module # Import for accessing module-level variables like 'flag'
from unittest.mock import patch, MagicMock # mock_open might not be needed if using BytesIO directly for client.post

# Define the allowed extensions based on services/load_process_docs.py for clarity,
# though this definition is not directly used by the imported allowed_file function.
ACTUAL_ALLOWED_EXTENSIONS_IN_APP = {'pdf', 'txt', 'csv', 'png', 'jpg'}

class TestApp(unittest.TestCase):

    # Removed @patch.dict from here as it doesn't solve import-time KeyError
    def setUp(self):
        """Set up test client and other test variables."""
        app.testing = True
        self.client = app.test_client()
        self.upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

        # Reset flag before each test that might modify it
        app_module.flag = None

    def tearDown(self):
        """Clean up after tests."""
        # Clean up any files created in the UPLOAD_FOLDER if necessary
        # This might not be strictly needed if file creation is properly mocked
        for f_name in os.listdir(self.upload_folder):
            try:
                os.remove(os.path.join(self.upload_folder, f_name))
            except OSError:
                pass # e.g. if it's a directory or access denied

    def test_allowed_file(self):
        """Test the allowed_file function based on app's actual ALLOWED_EXTENSIONS."""
        self.assertTrue(allowed_file("document.pdf")) # in app's list
        self.assertTrue(allowed_file("image.png"))    # in app's list
        self.assertTrue(allowed_file("textfile.txt"))  # in app's list
        self.assertTrue(allowed_file("data.csv"))      # in app's list
        self.assertFalse(allowed_file("archive.docx")) # NOT in app's list
        self.assertFalse(allowed_file("archive.doc"))  # NOT in app's list
        self.assertFalse(allowed_file("image.jpeg"))   # NOT in app's list (jpg is, not jpeg)
        self.assertFalse(allowed_file("script.py"))
        self.assertFalse(allowed_file("document.pdf.zip"))
        self.assertFalse(allowed_file("nodotextension"))
        self.assertTrue(allowed_file("UPPERCASE.PDF")) # in app's list
        self.assertTrue(allowed_file("mixed.CaSe.JpG"))  # jpg is in app's list

    @patch('chatbot.my_project.app.process_and_store_documents')
    @patch('werkzeug.datastructures.FileStorage.save') # Mock the save method of FileStorage
    @patch('chatbot.my_project.app.secure_filename')
    def test_upload_files_success(self, mock_secure_filename, mock_file_save, mock_process_and_store):
        """Test the /upload route with allowed files."""
        app_module.flag = None # Reset flag

        file_content = b"dummy pdf content"
        test_filename = "test.pdf" # Allowed extension

        # This is what app.py's os.path.join(UPLOAD_FOLDER, filename) would produce.
        # This path is passed to file.save() and then to process_and_store_documents.
        # UPLOAD_FOLDER is app.config['UPLOAD_FOLDER']
        expected_save_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))

        mock_secure_filename.return_value = test_filename

        file_data = io.BytesIO(file_content)
        response = self.client.post("/upload",
                                     content_type='multipart/form-data',
                                     data={'documents': (file_data, test_filename)})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Documents uploaded and processed.", response.get_data(as_text=True))

        # Check that secure_filename was called
        mock_secure_filename.assert_called_once_with(test_filename)

        # Check that file.save was called with the correct path
        mock_file_save.assert_called_once_with(expected_save_path)

        # Check that process_and_store_documents was called with the correct path
        mock_process_and_store.assert_called_once_with([expected_save_path])

        # Check that the flag is set
        # To test this, we need to access the 'flag' variable in the app module
        self.assertEqual(app_module.flag, 1)


    @patch('chatbot.my_project.app.process_and_store_documents')
    def test_upload_files_not_allowed(self, mock_process_and_store):
        """Test the /upload route with not allowed files."""
        # Using io.BytesIO for the file data
        file_data = io.BytesIO(b"dummy python content")
        response = self.client.post("/upload",
                                     content_type='multipart/form-data',
                                     data={'documents': (file_data, "test.py")})

        self.assertEqual(response.status_code, 200) # The route itself returns 200
        self.assertIn("Documents uploaded and processed.", response.get_data(as_text=True))
        # Even if no valid files are processed, the message is the same.
        # process_and_store_documents should be called with an empty list if no files are saved.
        mock_process_and_store.assert_called_once_with([])
        # According to current app.py logic, flag is set to 1 even if no valid files are uploaded.
        self.assertEqual(app_module.flag, 1)

    @patch('chatbot.my_project.app.answer_query_chatbot')
    def test_chat_route_no_retrieval(self, mock_answer_query_chatbot):
        """Test the /chat route when flag is not 1 (no retrieval)."""
        app_module.flag = None # Ensure flag is not 1
        app_module.session_history = [] # Clear session history

        mock_answer_query_chatbot.return_value = "Test bot response"

        response = self.client.post("/chat", json={"message": "Hello bot"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"response": "Test bot response"})
        mock_answer_query_chatbot.assert_called_once()
        # Check session history (simplified)
        self.assertTrue(any(isinstance(msg, app_module.HumanMessage) and msg.content == "Hello bot" for msg in app_module.session_history))
        self.assertTrue(any(isinstance(msg, app_module.AIMessage) and msg.content == "Test bot response" for msg in app_module.session_history))

    @patch('chatbot.my_project.app.retrieve_docs')
    def test_chat_route_with_retrieval(self, mock_retrieve_docs):
        """Test the /chat route when flag is 1 (with retrieval)."""
        app_module.flag = 1 # Set flag to 1 for retrieval
        app_module.session_history = []

        mock_retrieve_docs.return_value = ["doc1", "doc2"]

        # This tests the retrieval part. The current app.py code for flag == 1
        # doesn't then go on to call answer_query_chatbot or return its response.
        # It just appends to session_history and implicitly returns None (which Flask turns into a 200 OK with empty body).
        # This might be a bug in app.py or intended behavior for a specific workflow.
        # The test will reflect the current behavior of app.py.
        response = self.client.post("/chat", json={"message": "Retrieve context please"})

        self.assertEqual(response.status_code, 200)
        # Depending on Flask version and if the route truly returns nothing but appends to a list,
        # the response might be empty or contain a default HTML page if not returning jsonify.
        # The current app.py code path for `if flag == 1:` does not return a value from the function.
        # A view function in Flask should always return a response. If it returns None, Werkzeug
        # will raise a TypeError. This indicates a potential bug in the app.py's /chat route.
        # For now, let's assume it should return something, or the test needs to expect an error.
        # Given the current app.py, it will raise an error.
        # Let's assume the intention was to return the retrieved docs or a message.
        # If the route is fixed to `return jsonify({"retrieved_context": retrieved_docs})`
        # then the following assertions would be valid:
        # self.assertEqual(response.json, {"retrieved_context": ["doc1", "doc2"]})
        # mock_retrieve_docs.assert_called_once_with("Retrieve context please")
        # self.assertEqual(app_module.session_history, [{"retrieved_context": ["doc1", "doc2"]}])

        # Based on current app.py (June 19, 2024), the route will raise a TypeError.
        # So, a more accurate test for the *current* buggy code would be to expect an error.
        # However, unit tests should ideally test the *intended* behavior.
        # I will write the test assuming the route *should* return the retrieved docs.
        # This means I'm assuming a bug fix in app.py like:
        #
        # if flag == 1:
        #     retrieved_docs = retrieve_docs(user_message)
        #     session_history.append({"retrieved_context":retrieved_docs})
        #     return jsonify({"retrieved_context": retrieved_docs}) # ADDED THIS LINE FOR THE TEST
        #
        # If this fix is not made, this test will fail due to TypeError or 500 error.
        # For now, to make the test pass *without* modifying app.py and highlighting the issue:
        # The route *will* raise an exception because a view didn't return a response.
        # A test client call that results in an unhandled exception in the app
        # usually results in the HTML for a 500 error.

        # Given the current app.py, the path for flag == 1 leads to an error
        # because it doesn't return a response object.
        # NOW FIXED: The route should return jsonify({"retrieved_docs": retrieved_docs})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"retrieved_docs": ["doc1", "doc2"]})
        mock_retrieve_docs.assert_called_once_with("Retrieve context please")
        # Check session_history as before
        self.assertTrue(any(entry.get("retrieved_context") == ["doc1", "doc2"] for entry in app_module.session_history if isinstance(entry, dict)))

    def test_chat_route_no_message(self):
        """Test the /chat route with no message provided."""
        app_module.flag = None
        app_module.session_history = []
        response = self.client.post("/chat", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No message provided."})

    @patch('chatbot.my_project.app.answer_query_chatbot')
    def test_chat_route_exception_in_chatbot(self, mock_answer_query_chatbot):
        """Test the /chat route when answer_query_chatbot raises an exception."""
        app_module.flag = None
        app_module.session_history = []
        mock_answer_query_chatbot.side_effect = Exception("Chatbot failed")

        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {
            "response": "⚠️ Sorry, an unexpected error occurred.",
            "error": "Chatbot failed"
        })


if __name__ == '__main__':
    unittest.main()
