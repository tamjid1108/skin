# Skincare Sage

Skincare Sage is an intelligent skincare advisor that leverages Deep Learning to predict skin parameters such as acne severity and skin type based on a user's image input. Additionally, users can input other skin concerns, and Skincare Sage will provide personalized recommendations for essential skincare products tailored to their needs.

## Features

- **Image Analysis**: Skincare Sage analyzes user-provided images to predict acne severity and skin type.
- **Personalized Recommendations**: Based on the predicted skin parameters and user-inputted concerns, Skincare Sage suggests essential skincare products.
- **User Interaction**: Users can input additional skin concerns to receive more tailored recommendations.
- **Local Setup**: Skincare Sage can be easily set up and run locally on your system for convenient usage.

## How to Run Locally

To run Skincare Sage locally on your system, follow these steps:

### Prerequisites

- Python 3.x installed on your machine.
- Pip package manager installed.

### Installation

1. Clone the Skincare Sage repository to your local machine:

```
git clone https://github.com/tamjid1108/skin.git
```

2. Navigate to the project directory:

```
cd skin
```

3. Install the required dependencies:

**(Optionally create and run in a virtual environment)**
```
python3 -m venv myenv
```
  - **Windows**:
     ```bash
     myenv\Scripts\activate
     ```
  - **macOS/Linux**:
     ```bash
     source myenv/bin/activate
     ```
```
pip install -r requirements.txt
```

### Usage

1. Ensure you have a suitable image of your face for analysis or take one directly in the app.
2. Run the Skincare Sage script:

```
streamlit run app.py
```

3. Follow the on-screen prompts to upload your image and input any additional skin concerns.
4. Skincare Sage will provide you with personalized skincare recommendations based on the analysis results.


https://github.com/tamjid1108/skin/assets/76261405/e8253992-4661-4ef8-b503-ad986cc53a71


## Contributors

- [Tamjid](https://github.com/tamjid1108)
- [Jersha](https://github.com/jersha-heartly-x)

