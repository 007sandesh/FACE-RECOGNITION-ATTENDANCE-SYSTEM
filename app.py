import os
from flask import Flask, request, render_template, redirect, url_for, flash, session, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from datetime import date
from datetime import datetime
import mysql.connector
import hashlib
import pandas as pd
import numpy as np
import base64
import io
from PIL import Image
import json


app = Flask(__name__)
app.secret_key = "3fc18c5457e07839473ddd0a81dfff24"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # type: ignore

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(id):
    return User(id)

# MySQL database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "digitalhajir"
}

# Function to insert user login data into the database
def insert_user(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        hashed_password = hashlib.md5(password.encode()).hexdigest()
        sql = "INSERT INTO admin (username, password) VALUES (%s, %s)"
        values = (username, hashed_password)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error inserting user:", e)

# Function to check user login credentials
def check_user_credentials(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        provided_password = hashlib.md5(password.encode()).hexdigest()
        sql = "SELECT password FROM admin WHERE username = %s"
        values = (username,)
        cursor.execute(sql, values)
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        if user_data is not None:
            stored_password = user_data[0]
            if provided_password == stored_password:
                return True
        return False
    except Exception as e:
        print("Error checking user credentials:", e)
        return False

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# If these directories don't exist, create them
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
if f"Attendance-{datetoday}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{datetoday}.csv", "w") as f:
        f.write("Name,Roll,Time")

# Function to retrieve the total number of users from the database
def totalreg():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return total_users
    except Exception as e:
        print("Error fetching total number of users:", e)
        return 0  # Return 0 instead of None to avoid "None is not subscriptable" error

totalreg_count = totalreg()
if totalreg_count != 0:  # Changed from 'is not None'
    print("Total number of users:", totalreg_count)
else:
    print("No users found or failed to retrieve total number of users")

# Function to calculate Euclidean distance between feature vectors
def euclidean_distance(a, b):
    return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5

# The most common label among the k neighbors is returned as the prediction
def identify_face(facearray, k=5):
    faces = []
    labels = []
    
    # Check if static/faces directory exists
    if not os.path.exists("static/faces"):
        os.makedirs("static/faces")
        return "Unknown", 0.0
        
    userlist = os.listdir("static/faces")
    
    # If no users, return Unknown
    if not userlist:
        return "Unknown", 0.0
        
    for user in userlist:
        user_dir = os.path.join("static/faces", user)
        if not os.path.isdir(user_dir):
            continue
            
        user_images = os.listdir(user_dir)
        if not user_images:
            continue
            
        for imgname in user_images:
            try:
                img = Image.open(f"static/faces/{user}/{imgname}").convert("L").resize((50, 50))
                resized_face = np.array(img)
                faces.append(resized_face.flatten().tolist())
                labels.append(user)
            except Exception as e:
                print(f"Error loading image {imgname}: {str(e)}")
                continue
    
    # If no faces loaded, return Unknown
    if not faces:
        return "Unknown", 0.0
        
    test_face = facearray.flatten().tolist()
    distances = []
    for i, known_face in enumerate(faces):
        dist = euclidean_distance(test_face, known_face)
        distances.append((dist, labels[i]))
    
    distances.sort(key=lambda x: x[0])
    top_k = [label for _, label in distances[:min(k, len(distances))]]
    
    # Return the most common label among the k nearest neighbors
    if not top_k:
        return "Unknown", 0.0
        
    most_common_label = max(set(top_k), key=top_k.count)
    confidence = top_k.count(most_common_label) / len(top_k)
    return most_common_label, float(confidence)
# --- End Manual KNN Implementation ---

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    names = df["Name"]
    rolls = df["Roll"]
    times = df["Time"]
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    if isinstance(name, tuple):  # If name is a tuple (label, confidence)
        name = name[0]  # Extract just the label
    username = name.split("_")[0]
    userrollno = name.split("_")[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    if int(userrollno) not in list(df["Roll"]):
        with open(f"Attendance/Attendance-{datetoday}.csv", "a") as f:
            f.write(f"\n{username},{userrollno},{current_time}")

# Function to insert user details into the database
def insert_user_details(name, rollno, email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE rollno = %s OR email = %s", (rollno, email))
        existing_user = cursor.fetchone()
        if existing_user:
            if existing_user[1] == rollno:
                flash("Error: This roll number is already registered.", "danger")
            else:
                flash("Error: This email address is already taken.", "danger")
            return False
        sql = "INSERT INTO users (name, rollno, email) VALUES (%s, %s, %s)"
        values = (name, rollno, email)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print("Error inserting user details:", e)
        return False

# Function to fetch user details from the database
def get_user_details(id=None):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        if id is not None:
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (id,))
        else:
            sql = "SELECT * FROM users"
            cursor.execute(sql)
        user_details = cursor.fetchall()
        cursor.close()
        conn.close()
        return user_details
    except Exception as e:
        print("Error fetching user details:", e)
        return []

all_users = get_user_details()
user_details = get_user_details(id=1)

# Update user details in the database
def update_user_details(id, new_name, new_rollno, new_email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "UPDATE users SET name = %s, rollno = %s, email = %s WHERE id = %s"
        values = (new_name, new_rollno, new_email, id)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print("Error updating user details:", e)
    return False

# Function to retrieve user details by ID
def get_user_by_id(id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(sql, (id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Exception as e:
        print("Error fetching user details by ID:", e)
        return None

# Function to delete a user by their ID
def delete_user_by_id(id):
    try:
        # First get the user details before deleting
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get user details
        select_sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(select_sql, (id,))
        user = cursor.fetchone()
        
        if user:
            # Delete from database
            delete_sql = "DELETE FROM users WHERE id = %s"
            cursor.execute(delete_sql, (id,))
            conn.commit()
            
            # Delete user's face images directory
            if len(user) >= 3:  # Make sure we have name and rollno
                username = user[1]  # name
                rollno = user[2]   # rollno
                user_folder = f"static/faces/{username}_{rollno}"
                if os.path.exists(user_folder) and os.path.isdir(user_folder):
                    import shutil
                    shutil.rmtree(user_folder)
                    print(f"Deleted user folder: {user_folder}")
        
        cursor.close()
        conn.close()
        flash("User deleted successfully", "success")
    except Exception as e:
        flash(f"Failed to delete user: {str(e)}", "danger")
        print(f"Error in delete_user_by_id: {str(e)}")

# ROUTING FUNCTIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_user_credentials(username, password):
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            message = 'Invalid username or password'
    return render_template('login.html', message=message)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route("/home")
# @login_required
def home():
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route("/use")
def use():
    return render_template("use.html")

@app.route("/notrain")
def notrain():
    return render_template("notrain.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    if not data:  # Check if data is None to avoid "get is not a known attribute of None" error
        return jsonify({"error": "No JSON data received"}), 400
        
    try:
        image_data = data.get("image", "")
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((50, 50))
        np_image = np.array(image)

        label, confidence = identify_face(np_image)
        add_attendance(label)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/startuser")
def startuser():
    # No model file needed, just check if faces exist
    if not totalreg():
        return redirect("/notrain")
    return redirect("/detect")

@app.route("/startadmin", methods=["GET", "POST"])
@login_required
def startadmin():
    if not os.listdir("static/faces"):
        mess = ""
        return render_template("home.html", totalreg=totalreg(), datetoday2=datetoday2, mess="There are no faces in the static/faces folder. Please add a new face to continue.")
    
    # Redirect to detect page for admins as well, since we've removed OpenCV functionality
    return redirect("/detect")

@app.route("/add", methods=["GET", "POST"])
@login_required
def add():
    if request.method == "POST":
        newusername = request.form.get("newusername")
        newuserrollno = request.form.get("newuserrollno")
        useremail = request.form.get("newuseremail")
        if newusername and newuserrollno and useremail:
            if insert_user_details(newusername, newuserrollno, useremail):
                # Create a directory for the new user
                userimagefolder = "static/faces/" + newusername + "_" + str(newuserrollno)
                if not os.path.isdir(userimagefolder):
                    os.makedirs(userimagefolder)
                
                flash("User added successfully. Please capture face images on the next page.", "success")
                # Redirect to a page where JavaScript can capture face images
                return redirect(url_for("capture_faces", username=newusername, rollno=newuserrollno))
        else:
            flash("Please fill in all the fields", "danger")
    return render_template("add.html")

@app.route("/capture_faces/<username>/<rollno>")
@login_required
def capture_faces(username, rollno):
    return render_template("capture_faces.html", username=username, rollno=rollno)

@app.route("/save_face", methods=["POST"])
@login_required
def save_face():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
            
        username = data.get("username")
        rollno = data.get("rollno")
        image_data = data.get("image")
        index = data.get("index", 0)
        training_mode = data.get("trainingMode", False)
        
        # Create directory for the user if it doesn't exist
        user_dir = f"static/faces/{username}_{rollno}"
        if not os.path.isdir(user_dir):
            os.makedirs(user_dir)
        
        # Save the image
        image_data = image_data.split(",")[1]  # Remove the data URL prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save the image with index in filename
        filename = f"{user_dir}/face_{index}.jpg"
        image.save(filename)
        
        # If training mode is enabled, save additional metadata
        if training_mode:
            # Create a training data directory if it doesn't exist
            training_dir = f"{user_dir}/training_data"
            if not os.path.isdir(training_dir):
                os.makedirs(training_dir)
            
            # Save metadata about this training image
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "filename": f"face_{index}.jpg",
                "training_enabled": True
            }
            
            with open(f"{training_dir}/metadata_{index}.json", "w") as f:
                json.dump(metadata, f)
        
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving face:", e)
        return jsonify({"success": False, "error": str(e)})

@app.route("/save_training_data", methods=["POST"])
@login_required
def save_training_data():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
            
        username = data.get("username")
        rollno = data.get("rollno")
        
        # Create directory structure
        user_dir = f"static/faces/{username}_{rollno}"
        training_dir = f"{user_dir}/training_data"
        
        if not os.path.isdir(training_dir):
            os.makedirs(training_dir)
        
        # Save a summary file with timestamp
        training_summary = {
            "username": username,
            "rollno": rollno,
            "timestamp": datetime.now().isoformat(),
            "training_completed": True
        }
        
        with open(f"{training_dir}/training_summary.json", "w") as f:
            json.dump(training_summary, f)
        
        return jsonify({"success": True})
    except Exception as e:
        print("Error saving training data:", e)
        return jsonify({"success": False, "error": str(e)})

@app.route("/userdetails")
@login_required
def userdetails():
    user_details = get_user_details()
    return render_template("userdetails.html", user_details=user_details)

@app.route("/edit_user/<int:id>", methods=["GET", "POST"])
@login_required
def edit_user(id):
    if request.method == "POST":
        new_name = request.form["new_name"]
        new_rollno = request.form["new_rollno"]
        new_email = request.form["new_email"]
        if update_user_details(id, new_name, new_rollno, new_email):
            flash("User details updated successfully", "success")
            return redirect(url_for("userdetails"))
        else:
            flash("Failed to update user details", "danger")
    user = get_user_details(id)
    if user is None:
        flash("User not found", "danger")
        return redirect(url_for("userdetails"))
    return render_template("edit_user.html", user=user)

@app.route("/edit_user/<int:id>/update", methods=["POST"])
@login_required
def update_user(id):
    new_name = request.form["new_name"]
    new_rollno = request.form["new_rollno"]
    new_email = request.form["new_email"]
    if update_user_details(id, new_name, new_rollno, new_email):
        flash("User details updated successfully", "success")
    else:
        flash("Failed to update user details", "danger")
    return redirect(url_for("userdetails"))

@app.route("/delete_user_confirmation/<int:id>", methods=["GET"])
@login_required
def delete_user_confirmation(id):
    user = get_user_by_id(id)
    return render_template("delete_user_confirmation.html", user=user)

@app.route("/delete_user/<int:id>", methods=["POST"])
@login_required
def delete_user(id):
    user = get_user_details(id)
    if user is None:
        flash("User not found", "danger")
    else:
        delete_user_by_id(id)
        flash("User deleted successfully", "success")
    return redirect(url_for("userdetails"))

@app.route("/cleanup", methods=["GET"])
@login_required
def cleanup_folders():
    try:
        # Get all users from database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT name, rollno FROM users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Create a set of valid user folders
        valid_folders = set()
        for user in users:
            if len(user) >= 2:
                name = user[0]
                rollno = user[1]
                valid_folders.add(f"{name}_{rollno}")
        
        # Check all folders in static/faces directory
        face_folders = []
        orphaned_folders = []
        if os.path.exists("static/faces"):
            face_folders = [f for f in os.listdir("static/faces") if os.path.isdir(os.path.join("static/faces", f))]
            
        for folder in face_folders:
            if folder not in valid_folders:
                orphaned_folders.append(folder)
                folder_path = os.path.join("static/faces", folder)
                import shutil
                shutil.rmtree(folder_path)
        
        if orphaned_folders:
            flash(f"Cleaned up {len(orphaned_folders)} orphaned folders: {', '.join(orphaned_folders)}", "success")
        else:
            flash("No orphaned folders found", "info")
        
        return redirect(url_for("home"))
    except Exception as e:
        flash(f"Error cleaning up folders: {str(e)}", "danger")
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
