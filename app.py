import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

classification_model = joblib.load('classification_model.pkl')
regression_model = joblib.load('regression_model.pkl')
preprocessor = joblib.load("classification_preprocessor.pkl")

df = pd.read_csv('clean_data.csv')

def about(): #Home Page
    st.title("OpenLearn 1.0 Capstone Project\n# ID: OL-25-LP-059")
    st.header("Dataset Overview")
    st.markdown("""
    #### Dataset Source: [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    #### Collected by OSMI (Open Sourcing Mental Illness)
    #### Features include:
    * Demographic details (age, gender, country)
    * Workplace environment (mental health benefits, leave policies)
    * Personal experiences (mental illness, family history)
    * Attitudes towards mental health
    """)

    st.subheader("Problem Statement")
    st.markdown("""
    As a Machine Learning Engineer at NeuronInsights Analytics, you've been contracted by a coalition of
    leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
    disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
    proactively identify and support at-risk employees. Your role is to analyze survey data from over 1,500 tech
    professionals, covering workplace policies, personal mental health history, openness to seeking help, and
    perceived employer support.
                
    ### Project Objectives:
    * **Exploratory Data Analysis**
    * **Supervised Learning**:
        * *Classification task*: Predict whether a person is likely to seek mental health treatment (treatment column: yes/no)
        * *Regression task*: Predict the respondent's age
    * **Unsupervised Learning**: Cluster tech workers into mental health personas
    * **Streamlit App Deployment**
    """)

def visualization(): #Data Visualization Page
    st.title("Dataset Visualization")
    st.markdown("I used Matplotlib and Seaborn to visualize different features of the dataset to infer trends for them. I've performed univariate and Bivariate analysis on different features.")
    st.header("Features Used")
    st.code('''
    Age
    Gender
    self_employed
    family_history
    work_interfere
    no_employees
    remote_work
    tech_company
    benefits
    care_options
    wellness_program
    seek_help
    anonymity
    leave
    mental_health_consequence
    phys_health_consequence
    coworkers
    supervisor
    mental_health_interview
    phys_health_interview
    mental_vs_physical
    obs_consequence
    ''') 
    st.divider()
    st.header("Univariate Analysis")
    st.image("assets/Screenshot 2025-07-31 102942.png")
    st.subheader("Trends in Univariate Analysis")
    st.markdown('''
    1. **Age** - Most of people are middle aged, 30s is the most significant age group.

    2. **Gender** - Majority is Male(more than 78%), around 20% females and a little proportion of other genders conmbined.

    3. **Treatment Seeking** - Mixed Reviews, treatment seeking people are slightly more than those who don't.

    4. **Work Interference** - Most of the people lie in the 'Sometimes' cateogories, other cateogories have similar counts, which are much lower than it.

    5. **Family History** - There are more people who don't have family history of mental illness than those who have.

    6. **Coworker Openess** - Most people said that they are open to discuss mental health issue but only with some of the coworkers, there are very less number of people who completely agree or disagree with sharing a mental health issue.

    7. **Leave Grant** - Most people said that they don't know about how easy it is to get a medical leave for mental health condition, then there are those who said its somewhat easy, after that there are people who said it's very easy, then there are thos who said it's somewhat difficult, then those who said its very difficult to get a medical leave for mental health condition.

    8. **Benefits** - Many people agree that they are provided mental health benefits, also there is a significant group of people who don't know about it and there are also people significant in number who tell that they are not provided such benefits.
    ''')
    st.divider()
    st.header("Bivariate Analysis")
    st.image("assets/Screenshot 2025-08-01 001844.png")
    st.image("assets/Screenshot 2025-08-01 001855.png")
    st.subheader("Trends in Bivariate Analysis")
    st.markdown('''
    ==> There are more people who seek treatment in the 'Often' and 'Rarely' cateogory in work interference compared to those who don't. In 'Never' category people don't seeking treatment are much higher in number. In 'Sometimes' both categories' people are almost equal in number.

    ==> There are more people who don't seek treatment compared to those who seek till 30's age group. After that the trend gets reversed.

    ==> There are more males who don't seek treatment compared to those who do. But in female and other gender cateogories people seeking treatment are greater in number.

    ==> Gender distribution by work interference follows a normal trend as males are more in number so they can be seen in higher numbers in each cateogory. 'Sometimes' is the category of work interference which have most people from each gender cateogory.

    ==> Age distribution by gender also follows a normal trend. All gender cateogories have highest number of middle-aged people(30s).
    ''')
    

def supervised(): #Superivised Tasks Page
    st.title('A. Classification Task')
    st.header('About the Model')
    st.markdown('Classification is done on the basis of "treatment" column. Its a binary column so the datapoints are classified in the Yes or No categories')
    st.markdown(""" 
    I've made a grid of multiple algorithms and their hyperparameters, used GridSearchCV to find the best model.
    The algorithms which I've used are:
    - `Logistic Regression`
    - `XG Boost Classification`
    - `Support Vector Classification`
    - `K Nearest Neighbours Classification`
    - `Random Forest Classification`

    The judgement was made on the basis of F1 score. The XGB model stood best with F1score = 0.75.
    I also used a very basic pipeline to encode the numeric and cateogorical features efficently.
    """)
    st.subheader('Performance Comparison')
    st.code('''
    LogReg Best Params: {'C': 0.1}
    LogReg Best CV Score: 0.725378382792604

    SVC Best Params: {}
    SVC Best CV Score: 0.7072279017375422

    KNN Best Params: {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}
    KNN Best CV Score: 0.6274708357687999

    RF Best Params: {'max_depth': None, 'n_estimators': 200}
    RF Best CV Score: 0.7284325211040776

    XGB Best Params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
    XGB Best CV Score: 0.752083385831377


    Best Model: XGB with F1 = 0.752083385831377
    ''')
    st.subheader('Confusion Matrix')
    st.image("assets/Screenshot 2025-07-31 174750.png")
    st.subheader('Model Report')
    st.code('''
    precision    recall  f1-score   support

           0       0.76      0.78      0.77       115
           1       0.81      0.79      0.80       135

    accuracy                           0.78       250
   macro avg       0.78      0.78      0.78       250
weighted avg       0.78      0.78      0.78       250

Accuracy: 78.400%
ROC-AUC Score: 81.201%
    ''')

    st.subheader('Conclusion :-')
    st.markdown('''
    The XG Boost classifier model gave the best results, this can be inferred by comparing the scores of all models.


    F1 = 0.752083385831377


    Accuracy: 78.400%


    ROC-AUC Score: 81.201%
    ''')

    st.divider()

    st.title('B. Regression Task')
    st.header('About the Model')
    st.markdown('In regression task the objective was to predict the age. To get the best prediction, I made a grid of multiple algorithms and judged their performance on the basis of R2 Score.')
    st.subheader('Performance Comparison')
    st.code("""
    Ridge Best Params: {'alpha': 100}
    Ridge Best Score: 0.07090437207291325

    Lasso Best Params: {'alpha': 0.1}
    Lasso Best Score: 0.06445819816270804

    Elastic Best Params: {'alpha': 0.1, 'l1_ratio': 0.3}
    Elastic Best Score: 0.07078298709601656

    SVR Best Params: {}
    SVR Best Score: 0.013222759990994227

    RF Best Params: {'max_depth': 7, 'n_estimators': 100}
    RF Best Score: 0.040791720357292525

    KNN Best Params: {'n_neighbors': 9, 'p': 2, 'weights': 'uniform'}
    KNN Best Score: -0.024845999894550785

    XGB Best Params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}
    XGB Best Score: 0.06743408441543579


    Best Model: Ridge with R2 = 0.07090437207291325
    """)
    st.subheader('Actual v/s Predicted')
    st.image("assets/Screenshot 2025-07-31 173408.png")
    st.subheader("Conclusion")
    st.markdown('''
    The Ridge model gave the best results, this can be inferred by comparing the scores of all models. The dataset was not so good for regression, that's why we are getting such low scores in regression for all models.
                

    R2_Score:  0.05189627534332486
                

    MAE:  5.4122502689239536
                

    RMSE:  7.042636876899095
    ''')

def unsupervised(): #Unsuperivised Task Page
    st.title("Unsupervised Task")
    st.divider()
    st.header("About the Model")
    st.markdown("For the clustering task I used the most relevant features like treatment, work_interfere etc. I used t-SNE to for dimension reduction and clear visualization of the clusters. I have compared the K-Means and Agglomerative Clustering algorithms for different values of n_clusters and judged them on the basis of Silhouette score. I found out that using K-Means with n_clusters = 4 will be ideal for the task.")
    st.divider()
    st.header("Clusters Visualization")
    st.image("assets/Screenshot 2025-07-30 194206.png")
    st.divider()
    st.header("Clusters description")
    st.markdown("""
    I have identified 4 major clusters of all the available datapoints.  
    Each cluster represents a different mental health persona which can be understood from the following description:  

    ---

    ### **Persona 1 - Silent Sufferers**  
    This group often feels alone with their problems.  
    They don't think their workplace supports them and don't feel safe talking about mental health.  
    Their stress is high, but they rarely ask for help, even if they've faced issues before.  
    Most of them work in smaller companies where support is harder to find.  

    ---

    ### **Persona 2 - Open Advocates**  
    These are the people who are open about mental health and encourage others to talk about it too.  
    They know what help is available at work and use it when they need to.  
    Many work in bigger or hybrid companies where talking about mental health is more common.  

    ---

    ### **Persona 3 - Under-Supported Professionals**  
    This group knows they are struggling but feels like their workplace doesn't help enough.  
    Their stress is high, and while they want to talk about it, they often don't get the support they need.  
    Many of them are in large companies or remote roles, where help exists on paper but doesn't really reach them.  

    ---

    ### **Persona 4 - Resilient Optimists**  
    These people are generally doing well mentally.  
    They feel supported by their workplace, don't face much stress, and are open to talking about mental health if needed.  
    Many are younger or have flexible jobs, and they often show what good support can look like.  
    """)

def classifier_demo():
    st.title("Mental Health Treatment Classifier")

    age = st.slider("Age", 15, 70, 25)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary/Other'])
    self_employed = st.selectbox("Self Employed?", ['No', 'Yes'])
    family_history = st.selectbox("Family History of Mental Illness?", ['No', 'Yes'])
    work_interfere = st.selectbox("Work Interfere", ['Sometimes', 'Never', 'Rarely', 'Often'])
    no_employees = st.selectbox("No. of Employees", ['6-25', '26-100', 'More than 1000', '100-500', '1-5', '500-1000'])
    remote_work = st.selectbox("Remote Work?", ['No', 'Yes'])
    tech_company = st.selectbox("Tech Company?", ['Yes', 'No'])
    benefits = st.selectbox("Mental Health Benefits?", ['Yes', "Don't know", 'No'])
    care_options = st.selectbox("Care Options?", ['No', 'Yes', 'Not sure'])
    wellness_program = st.selectbox("Wellness Program?", ['No', 'Yes', "Don't know"])
    seek_help = st.selectbox("Seek Help?", ['No', "Don't know", 'Yes'])
    anonymity = st.selectbox("Anonymity?", ["Don't know", 'Yes', 'No'])
    leave = st.selectbox("Leave", ["Don't know", 'Somewhat easy', 'Very easy', 'Somewhat difficult', 'Very difficult'])
    mental_health_consequence = st.selectbox("Mental Health Consequence", ['No', 'Maybe', 'Yes'])
    phys_health_consequence = st.selectbox("Physical Health Consequence", ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox("Coworkers", ['Some of them', 'No', 'Yes'])
    supervisor = st.selectbox("Supervisor", ['Yes', 'No', 'Some of them'])
    mental_health_interview = st.selectbox("Mental Health Interview", ['No', 'Maybe', 'Yes'])
    phys_health_interview = st.selectbox("Physical Health Interview", ['Maybe', 'No', 'Yes'])
    mental_vs_physical = st.selectbox("Mental vs Physical", ["Don't know", 'Yes', 'No'])
    obs_consequence = st.selectbox("Observed Consequences", ['No', 'Yes'])

    input_dict = {
        'Age': age,
        'Gender': gender,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'no_employees': no_employees,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'phys_health_consequence': phys_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'phys_health_interview': phys_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence
    }

    features = pd.DataFrame([input_dict])
    if 'Age' in features.columns:
        features['Age_log'] = np.log1p(features['Age'])
        features.drop('Age', axis=1, inplace=True)
    else:
    st.error("Age field missing in input.")

    if st.button("Predict"):
        transformed = preprocessor.transform(features)
        prediction = classification_model.predict(transformed)
        if prediction[0] == 1:
            st.success("✅ Predicted: Will seek treatment!")
        else:
            st.error("❌ Predicted: Will not seek treatment!")

pg = st.navigation([
    st.Page(about, title="Welcome to the app!"),
    st.Page(visualization, title="Data Visualization"),
    st.Page(supervised, title="Classification and regression task"),
    st.Page(unsupervised, title="Unsupervised Task"),
    st.Page(classifier_demo, title="Classifier Demo")
])
pg.run()
