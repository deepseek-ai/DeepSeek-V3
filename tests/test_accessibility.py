import unittest
from flask import Flask, render_template_string
from flask_testing import TestCase
import hcaptcha

class TestAccessibility(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.secret_key = 'test_secret_key'

        hcaptcha_site_key = 'test_hcaptcha_site_key'
        hcaptcha_secret_key = 'test_hcaptcha_secret_key'

        @app.route('/signup', methods=['GET', 'POST'])
        def signup():
            if request.method == 'POST':
                hcaptcha_response = request.form['h-captcha-response']
                if hcaptcha.verify(hcaptcha_secret_key, hcaptcha_response):
                    return 'Sign-up successful!'
                else:
                    return 'hCaptcha verification failed.'
            return render_template_string('''
                <form method="POST">
                    <label for="username" aria-label="Username">Username</label>
                    <input type="text" id="username" name="username" required aria-required="true">
                    <label for="email" aria-label="Email">Email</label>
                    <input type="email" id="email" name="email" required aria-required="true">
                    <label for="password" aria-label="Password">Password</label>
                    <input type="password" id="password" name="password" required aria-required="true">
                    <div class="h-captcha" data-sitekey="{{ hcaptcha_site_key }}" role="presentation" aria-hidden="true"></div>
                    <button type="submit" aria-label="Sign Up">Sign Up</button>
                </form>
            ''', hcaptcha_site_key=hcaptcha_site_key)

        @app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                hcaptcha_response = request.form['h-captcha-response']
                if hcaptcha.verify(hcaptcha_secret_key, hcaptcha_response):
                    return 'Login successful!'
                else:
                    return 'hCaptcha verification failed.'
            return render_template_string('''
                <form method="POST">
                    <label for="username" aria-label="Username">Username</label>
                    <input type="text" id="username" name="username" required aria-required="true">
                    <label for="password" aria-label="Password">Password</label>
                    <input type="password" id="password" name="password" required aria-required="true">
                    <div class="h-captcha" data-sitekey="{{ hcaptcha_site_key }}" role="presentation" aria-hidden="true"></div>
                    <button type="submit" aria-label="Login">Login</button>
                </form>
            ''', hcaptcha_site_key=hcaptcha_site_key)

        return app

    def test_signup_accessibility(self):
        response = self.client.get('/signup')
        self.assert200(response)
        self.assertIn(b'aria-label="Username"', response.data)
        self.assertIn(b'aria-label="Email"', response.data)
        self.assertIn(b'aria-label="Password"', response.data)
        self.assertIn(b'aria-hidden="true"', response.data)

    def test_login_accessibility(self):
        response = self.client.get('/login')
        self.assert200(response)
        self.assertIn(b'aria-label="Username"', response.data)
        self.assertIn(b'aria-label="Password"', response.data)
        self.assertIn(b'aria-hidden="true"', response.data)

if __name__ == '__main__':
    unittest.main()
