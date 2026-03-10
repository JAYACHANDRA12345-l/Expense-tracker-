# Expense-tracker-

from flask import Flask, render_template_string, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------------
# Database Model
# ------------------------
class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Create Database
with app.app_context():
    db.create_all()

# ------------------------
# HTML Template
# ------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Expense Tracker </title>
<style>
body {
    font-family: Arial;
    background: url('https://th.bing.com/th/id/OIP.El6yCvVSRvFg6CmBve2sOAHaEo?w=261&h=180&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3') no-repeat center center fixed;
    background-size: cover;
    color: white;
    padding: 20px;
}
input,select,button {padding:10px;margin:5px;border-radius:5px;border:none;}
button {background:#ff6b6b;color:white;}
table {
    background: rgba(255,255,255,0.9);
    color: black;
    width: 100%;
    margin-top: 20px;
    border-radius: 10px;
}
td,th {padding:10px;}
a {color:yellow;}
</style>
</head>
<body>

<h1>Expense Tracker </h1>
<h3>Total: ₹{{total}}</h3>

<form method="POST" action="/add">
<input name="description" placeholder="Description" required>
<input name="amount" type="number" step="0.01" placeholder="Amount" required>

<select name="category">
<option>Food</option>
<option>Transport</option>
<option>Shopping</option>
<option>Entertainment</option>
<option>Other</option>
</select>

<button type="submit">Add</button>
</form>

<br>

<a href="/chart"><button>View Chart</button></a>
<a href="/predict"><button>Predict Future Expense</button></a>

<table border="1">
<tr>
<th>Description</th>
<th>Amount</th>
<th>Category</th>
<th>Date</th>
<th>Delete</th>
</tr>

{% for e in expenses %}
<tr>
<td>{{e.description}}</td>
<td>{{e.amount}}</td>
<td>{{e.category}}</td>
<td>{{e.date.strftime('%Y-%m-%d')}}</td>
<td><a href="/delete/{{e.id}}">Delete</a></td>
</tr>
{% endfor %}
</table>

{% if chart %}
<h2>Chart</h2>
<img src="/static/chart.png">
{% endif %}

{% if prediction %}
<h2>{{prediction}}</h2>
{% endif %}

</body>
</html>
"""

# ------------------------
# Home Route
# ------------------------
@app.route('/')
def index():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    total = sum(e.amount for e in expenses)

    return render_template_string(
        HTML,
        expenses=expenses,
        total=total,
        chart=False,
        prediction=None
    )

# ------------------------
# Add Expense
# ------------------------
@app.route('/add', methods=['POST'])
def add():
    desc = request.form['description']
    amount = float(request.form['amount'])
    category = request.form['category']

    expense = Expense(
        description=desc,
        amount=amount,
        category=category
    )

    db.session.add(expense)
    db.session.commit()

    return redirect('/')

# ------------------------
# Delete Expense
# ------------------------
@app.route('/delete/<int:id>')
def delete(id):
    expense = Expense.query.get(id)
    db.session.delete(expense)
    db.session.commit()
    return redirect('/')

# ------------------------
# Chart Route
# ------------------------
@app.route('/chart')
def chart():
    expenses = Expense.query.all()
    if not expenses:
        return redirect('/')

    data = {
        "category": [e.category for e in expenses],
        "amount": [e.amount for e in expenses]
    }

    df = pd.DataFrame(data)
    summary = df.groupby("category").sum().reset_index()

    if not os.path.exists("static"):
        os.mkdir("static")

    plt.figure(figsize=(6,4))
    sns.barplot(x="category", y="amount", data=summary)
    plt.title("Expense by Category")
    plt.tight_layout()
    plt.savefig("static/chart.png")
    plt.close()

    total = sum(e.amount for e in expenses)

    return render_template_string(
        HTML,
        expenses=expenses,
        total=total,
        chart=True,
        prediction=None
    )

# ------------------------
# Prediction Route
# ------------------------
@app.route('/predict')
def predict():
    expenses = Expense.query.order_by(Expense.date).all()
    if len(expenses) < 2:
        return redirect('/')

    days = np.array(
        [(e.date - expenses[0].date).days for e in expenses]
    ).reshape(-1,1)

    amounts = np.array([e.amount for e in expenses])

    model = LinearRegression()
    model.fit(days, amounts)

    future_day = np.array([[days[-1][0] + 30]])
    prediction = model.predict(future_day)[0]

    total = sum(e.amount for e in expenses)

    return render_template_string(
        HTML,
        expenses=expenses,
        total=total,
        chart=False,
        prediction=f"Predicted expense after 30 days: ₹{prediction:.2f}"
    )

# ------------------------
# Run App
# ------------------------
if __name__ == '__main__':
    app.run(debug=True)
