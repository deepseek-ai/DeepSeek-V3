from flask import Flask, render_template, request, redirect, url_for, flash
import hcaptcha

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure hCaptcha
hcaptcha_site_key = 'your_hcaptcha_site_key'
hcaptcha_secret_key = 'your_hcaptcha_secret_key'

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        hcaptcha_response = request.form['h-captcha-response']
        if hcaptcha.verify(hcaptcha_secret_key, hcaptcha_response):
            # Process the sign-up form
            flash('Sign-up successful!', 'success')
            return redirect(url_for('login'))
        else:
            flash('hCaptcha verification failed. Please try again.', 'danger')
    return render_template('signup.html', hcaptcha_site_key=hcaptcha_site_key)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        hcaptcha_response = request.form['h-captcha-response']
        if hcaptcha.verify(hcaptcha_secret_key, hcaptcha_response):
            # Process the login form
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('hCaptcha verification failed. Please try again.', 'danger')
    return render_template('login.html', hcaptcha_site_key=hcaptcha_site_key)

@app.route('/dashboard')
def dashboard():
    return 'Welcome to the dashboard!'

if __name__ == '__main__':
    app.run(debug=True)
