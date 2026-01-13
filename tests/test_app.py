import unittest
from app import app

class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)

    def test_predict_get(self):
        r = self.client.get('/predict')
        self.assertEqual(r.status_code, 200)

    def test_api_predict_missing(self):
        r = self.client.post('/api/predict', json={})
        self.assertEqual(r.status_code, 400)

if __name__ == '__main__':
    unittest.main()
