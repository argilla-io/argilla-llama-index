from argilla import init
from argilla_llama_index import ArgillaCallbackHandler

import unittest
from unittest import mock
from unittest.mock import Mock, MagicMock, patch

import requests_mock

class TestArgillaCallbackHandler(unittest.TestCase):

    @mock.patch('httpx.get')
    @mock.patch('argilla.init')  # Replace 'your_module.some_method1' with the actual method to patch
    #@mock.patch('argilla.datasets.push_to_argilla')  # Replace 'your_module.some_method2' with the actual method to patch
    def test_constructor_calls(self, mock_some_method1, mock_get):
        # Mock the HTTP request made by your method
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'key': 'value'}
        
        # Create an instance of ArgillaCallbackHandler
        handler = ArgillaCallbackHandler(
            dataset_name="dataset_name",
            workspace_name="workspace_name",
            api_url="http://localhost:6900",
            api_key="api_key",
        )

        # Assert that some_method1 was called with the expected parameters
        mock_some_method1.assert_called_once_with(api_key="api_key", api_url="http://localhost:6900")

        # Assert that some_method2 was called with the expected parameters
        #mock_some_method2.assert_called_once_with('expected_param_for_some_method2')


if __name__ == "__main__":
    unittest.main()
