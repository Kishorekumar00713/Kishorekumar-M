from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI

import os
import base64
import mimetypes


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# UPLOAD CONFIGURATION
# --------------------------------------------------

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------

client = OpenAI()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# AI ROOM ANALYSIS FUNCTION
# --------------------------------------------------

def analyze_room_with_ai(image_path, room_type, design_style, budget):

    try:

        # Get image MIME type
        mime_type, _ = mimetypes.guess_type(image_path)

        if mime_type is None:
            mime_type = "image/jpeg"


        # Read image
        with open(image_path, "rb") as image_file:

            image_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")


        # Create image data URL
        image_url = f"data:{mime_type};base64,{image_data}"


        # AI prompt
        prompt = f"""
You are an expert interior designer and room analysis assistant.

Analyze the uploaded room image carefully.

User requirements:

Room Type: {room_type}
Preferred Design Style: {design_style}
Budget: ₹{budget}

Analyze the ACTUAL uploaded room and provide a personalized interior
design recommendation.

Please include:

1. Current Room Analysis
   - approximate layout
   - existing furniture
   - wall colors
   - flooring
   - lighting
   - windows/doors
   - available open space

2. Problems or Areas That Can Be Improved

3. Recommended Color Palette

4. Furniture Recommendations

5. Lighting Recommendations

6. Wall and Ceiling Recommendations

7. Decor Recommendations

8. Space Utilization Suggestions

9. Budget-conscious recommendations suitable for the user's budget

10. A short final design concept explaining how the room should look
after the redesign.

Important:
- Base your observations on the actual image.
- Do not claim to detect objects that are not reasonably visible.
- Respect the selected design style.
- Keep recommendations realistic for the given budget.
- Do not recommend replacing everything if existing furniture can be reused.
"""


        # Send image + prompt to OpenAI
        response = client.responses.create(

            model="gpt-5.6-luna",

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",
                            "text": prompt
                        },

                        {
                            "type": "input_image",
                            "image_url": image_url
                        }

                    ]
                }
            ]
        )


        # Get AI response
        analysis = response.output_text

        return analysis


    except Exception as e:

        print("\n===== AI ERROR =====")
        print(str(e))

        return (
            "AI analysis could not be generated at this moment. "
            "Please check your API configuration and try again."
        )


# --------------------------------------------------
# GENERATE DESIGN
# --------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate():

    # Get user inputs
    room_type = request.form.get("room_type")

    design_style = request.form.get("design_style")

    budget = request.form.get("budget")


    # Get uploaded image
    image = request.files.get("room_image")


    print("\n================================")
    print("       DESIGN REQUEST")
    print("================================")

    print("Room Type:", room_type)
    print("Design Style:", design_style)
    print("Budget:", budget)


    # --------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------

    image_filename = None

    image_path = None


    if image and image.filename:

        image_filename = secure_filename(
            image.filename
        )


        image_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            image_filename
        )


        image.save(image_path)


        print("Image:", image_filename)

        print("Image saved at:", image_path)


    else:

        print("Image: No image uploaded")


    # --------------------------------------------------
    # BUDGET
    # --------------------------------------------------

    try:

        budget_value = float(budget)

    except (ValueError, TypeError):

        budget_value = 50000


    # --------------------------------------------------
    # AI ROOM ANALYSIS
    # --------------------------------------------------

    ai_analysis = None


    if image_path:

        print("\n===== AI ROOM ANALYSIS =====")

        ai_analysis = analyze_room_with_ai(

            image_path,

            room_type,

            design_style,

            budget_value
        )

        print("\nAI ANALYSIS:")

        print(ai_analysis)


    else:

        ai_analysis = (
            "No room image was uploaded. "
            "Please upload a room image for AI analysis."
        )


    # --------------------------------------------------
    # FALLBACK DESIGN DATA
    # --------------------------------------------------

    recommendations = {

        "Modern": {

            "colors": [
                "Warm White",
                "Light Grey",
                "Beige",
                "Wooden Brown"
            ],

            "furniture": [
                "Minimalist 3-seater sofa",
                "Modern coffee table",
                "Floating TV unit",
                "Accent chair"
            ],

            "lighting": [
                "Warm LED ceiling lights",
                "Pendant light",
                "Modern floor lamp"
            ],

            "decor": [
                "Indoor plants",
                "Abstract wall art",
                "Minimal curtains",
                "Decorative cushions"
            ]
        },


        "Minimalist": {

            "colors": [
                "White",
                "Light Beige",
                "Soft Grey",
                "Natural Wood"
            ],

            "furniture": [
                "Simple functional sofa",
                "Compact coffee table",
                "Storage cabinet",
                "Minimal shelving"
            ],

            "lighting": [
                "Recessed ceiling lights",
                "Warm LED strips",
                "Simple floor lamp"
            ],

            "decor": [
                "Small indoor plants",
                "Simple wall art",
                "Neutral curtains",
                "Minimal accessories"
            ]
        },


        "Traditional": {

            "colors": [
                "Cream",
                "Golden Beige",
                "Dark Brown",
                "Terracotta"
            ],

            "furniture": [
                "Wooden sofa set",
                "Traditional coffee table",
                "Wooden storage unit",
                "Decorative side tables"
            ],

            "lighting": [
                "Decorative chandelier",
                "Warm ceiling lights",
                "Traditional table lamps"
            ],

            "decor": [
                "Traditional paintings",
                "Decorative mirrors",
                "Indoor plants",
                "Patterned curtains"
            ]
        },


        "Luxury": {

            "colors": [
                "Ivory",
                "Charcoal Grey",
                "Gold",
                "Dark Wood"
            ],

            "furniture": [
                "Premium sectional sofa",
                "Marble coffee table",
                "Designer TV unit",
                "Accent chairs"
            ],

            "lighting": [
                "Designer chandelier",
                "Warm spotlights",
                "Decorative floor lamps"
            ],

            "decor": [
                "Large artwork",
                "Designer curtains",
                "Decorative mirrors",
                "Premium accessories"
            ]
        }

    }


    design = recommendations.get(

        design_style,

        recommendations["Modern"]
    )


    # --------------------------------------------------
    # BUDGET ALLOCATION
    # --------------------------------------------------

    furniture_budget = budget_value * 0.55

    lighting_budget = budget_value * 0.15

    decor_budget = budget_value * 0.10

    paint_budget = budget_value * 0.10

    miscellaneous_budget = budget_value * 0.10


    budget_breakdown = {

        "Furniture": round(furniture_budget),

        "Lighting": round(lighting_budget),

        "Decor": round(decor_budget),

        "Paint": round(paint_budget),

        "Miscellaneous": round(miscellaneous_budget)

    }


    # --------------------------------------------------
    # SEND DATA TO RESULT PAGE
    # --------------------------------------------------

    return render_template(

        "result.html",

        room_type=room_type,

        design_style=design_style,

        budget=budget_value,

        image_name=image_filename,

        design=design,

        budget_breakdown=budget_breakdown,

        ai_analysis=ai_analysis

    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)