from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

from src.logger import logger


# =========================================================
# FLASK APPLICATION
# =========================================================

application = Flask(__name__)

app = application


# =========================================================
# HOME PAGE
# =========================================================

@app.route('/')

def index():

    return render_template('index.html')


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route('/predictdata', methods=['GET', 'POST'])

def predict_datapoint():

    try:

        # -------------------------------------------------
        # GET REQUEST
        # -------------------------------------------------

        if request.method == 'GET':

            return render_template('home.html')

        # -------------------------------------------------
        # POST REQUEST
        # -------------------------------------------------

        else:

            logger.info(
                "Received Form Data"
            )

            # ---------------------------------------------
            # GET DATA FROM HTML FORM
            # ---------------------------------------------

            data = CustomData(

                gender=request.form.get(
                    'gender'
                ),

                race_ethnicity=request.form.get(
                    'race_ethnicity'
                ),

                parental_level_of_education=request.form.get(
                    'parental_level_of_education'
                ),

                lunch=request.form.get(
                    'lunch'
                ),

                test_preparation_course=request.form.get(
                    'test_preparation_course'
                ),

                reading_score=float(
                    request.form.get(
                        'reading_score'
                    )
                ),

                writing_score=float(
                    request.form.get(
                        'writing_score'
                    )
                )

            )

            logger.info(
                "Custom Data Object Created Successfully"
            )

            # ---------------------------------------------
            # CONVERT INPUT DATA TO DATAFRAME
            # ---------------------------------------------

            pred_df = data.get_data_as_data_frame()

            logger.info(
                f"Input DataFrame:\n{pred_df}"
            )

            print(
                "\nInput DataFrame:"
            )

            print(pred_df)

            # ---------------------------------------------
            # PREDICTION PIPELINE
            # ---------------------------------------------

            predict_pipeline = PredictPipeline()

            logger.info(
                "Prediction Pipeline Started"
            )

            # ---------------------------------------------
            # MAKE PREDICTION
            # ---------------------------------------------

            results = predict_pipeline.predict(
                pred_df
            )

            predicted_score = round(
                results[0],
                2
            )

            logger.info(
                f"Predicted Math Score: "
                f"{predicted_score}"
            )

            print(
                f"\nPredicted Math Score: "
                f"{predicted_score}"
            )

            # ---------------------------------------------
            # RETURN RESULT TO FRONTEND
            # ---------------------------------------------

            return render_template(

                'home.html',

                results=predicted_score

            )

    except Exception as e:

        logger.info(
            f"Exception Occurred: {e}"
        )

        print(e)

        return render_template(

            "home.html",

            results="Something went wrong"

        )


# =========================================================
# MAIN FUNCTION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

       
    )
    