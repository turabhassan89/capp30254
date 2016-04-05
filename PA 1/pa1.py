import pandas as pd
import urllib.request
import json
import matplotlib.pyplot as plt


genderize = 'https://api.genderize.io/?name='

def pa1():
    df = pd.read_csv('mock_student_data.csv')
    
    #Plotting the Histograms
    df.hist('Age')
    plt.savefig('Age')
    df.hist('GPA')
    plt.savefig('GPA')
    df.hist('Days_missed')
    plt.savefig('Days_missed')    
    
    #Finding the Summary Stats
    print( df.describe ( percentiles = [] ).transpose() )
    print( df.mode(numeric_only = True ) )
    print (df.isnull().sum())
    
    #Filling in the Missing values for Gender
    for index, row in df.iterrows():
        if pd.isnull( row['Gender'] ):
            name = row['First_name']
            url = genderize + name
            request = urllib.request.urlopen(url)
            data = json.loads(request.read().decode('utf8'))
            gender = data['gender'].capitalize()
            df.set_value(index,'Gender', gender) 
    
    #Filling in Missing Values Based on mean   
    df_mean = df.copy (deep = True)
    
    mean_age = df_mean['Age'].mean()
    df_mean['Age'].fillna ( mean_age, inplace = True )

    mean_gpa = df_mean['GPA'].mean()
    df_mean['GPA'].fillna ( mean_gpa, inplace = True )

    mean_daysmissed = df_mean['Days_missed'].mean()
    df_mean['Days_missed'].fillna ( mean_daysmissed, inplace = True )


    df_mean.to_csv('mean.csv')


    #Filling in missing values based on conditional mean
    df_conditional_mean = df.copy(deep = True)

    mean_age_grad_yes = df_conditional_mean['Age'][ df_conditional_mean['Graduated'] == 'Yes' ].mean()
    mean_age_grad_no = df_conditional_mean['Age'][ df_conditional_mean['Graduated'] == 'No' ].mean()
    
    mean_gpa_grad_yes = df_conditional_mean['GPA'][ df_conditional_mean['Graduated'] == 'Yes' ].mean()
    mean_gpa_grad_no = df_conditional_mean['GPA'][ df_conditional_mean['Graduated'] == 'No' ].mean()
    
    mean_daysmissed_grad_yes = df_conditional_mean['Days_missed'][ df_conditional_mean['Graduated'] == 'Yes' ].mean()
    mean_daysmissed_grad_no = df_conditional_mean['Days_missed'][ df_conditional_mean['Graduated'] == 'No' ].mean()

    for index, row in df_conditional_mean.iterrows():
        if pd.isnull(row['Age']) and row['Graduated'] == 'Yes':
            df_conditional_mean.set_value(index,'Age', mean_age_grad_yes)
        if pd.isnull(row['Age']) and row['Graduated'] == 'No':
            df_conditional_mean.set_value(index,'Age', mean_age_grad_no)

        if pd.isnull(row['GPA']) and row['Graduated'] == 'Yes':
            df_conditional_mean.set_value(index,'GPA', mean_gpa_grad_yes)
        if pd.isnull(row['GPA']) and row['Graduated'] == 'No':
            df_conditional_mean.set_value(index,'Age', mean_gpa_grad_no)
        
        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'Yes':
            df_conditional_mean.set_value(index,'Days_missed', mean_daysmissed_grad_yes)
        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'No':
            df_conditional_mean.set_value(index,'Days_missed', mean_daysmissed_grad_no)

    df_conditional_mean.to_csv('conditional_mean.csv')

    #Another way to filling in the missing values is to condition on two things; Gender and Graduation
    #Find the means for Male Graduated, Male Not Graduated, Female Graduated and Female Not Graduated
    #and fill in the missing values accordingly.        
            
    df_double_conditional = df.copy(deep = True)

    mean_age_grad_yes_male = df_double_conditional['Age'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender'] == 'Male') ].mean()
    mean_age_grad_no_male = df_double_conditional['Age'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender'] == 'Male') ].mean()
    mean_age_grad_yes_female = df_double_conditional['Age'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender'] == 'Female') ].mean()
    mean_age_grad_no_female = df_double_conditional['Age'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender'] == 'Female') ].mean()
    
    mean_gpa_grad_yes_male = df_double_conditional['GPA'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender'] == 'Male') ].mean()
    mean_gpa_grad_no_male = df_double_conditional['GPA'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender'] == 'Male' )].mean()
    mean_gpa_grad_yes_female = df_double_conditional['GPA'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender'] == 'Female') ].mean()
    mean_gpa_grad_no_female = df_double_conditional['GPA'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender'] == 'Female') ].mean()

    mean_daysmissed_grad_yes_male = df_double_conditional['Days_missed'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender' ] == 'Male' ) ].mean()
    mean_daysmissed_grad_no_male = df_double_conditional['Days_missed'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender' ] == 'Male') ].mean()
    mean_daysmissed_grad_yes_female = df_double_conditional['Days_missed'][ (df_double_conditional['Graduated'] == 'Yes') & (df_double_conditional['Gender'] == 'Female' )].mean()
    mean_daysmissed_grad_no_female = df_double_conditional['Days_missed'][ (df_double_conditional['Graduated'] == 'No') & (df_double_conditional['Gender'] == 'Female') ].mean()

    for index, row in df_double_conditional.iterrows():
        if pd.isnull(row['Age']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'Age', mean_age_grad_yes_male)        
        if pd.isnull(row['Age']) and row['Graduated'] == 'No' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'Age', mean_age_grad_no_male)
        if pd.isnull(row['Age']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'Age', mean_age_grad_yes_female)            
        if pd.isnull(row['Age']) and row['Graduated'] == 'No' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'Age', mean_age_grad_no_female)

        if pd.isnull(row['GPA']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'GPA', mean_gpa_grad_yes_male)
        if pd.isnull(row['GPA']) and row['Graduated'] == 'No' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'GPA', mean_gpa_grad_no_male)
        if pd.isnull(row['GPA']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'GPA', mean_gpa_grad_yes_female)
        if pd.isnull(row['GPA']) and row['Graduated'] == 'No' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'GPA', mean_gpa_grad_yes_female)

        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'Days_missed', mean_daysmissed_grad_yes_male)
        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'No' and row['Gender'] == 'Male':
            df_double_conditional.set_value(index,'Days_missed', mean_daysmissed_grad_no_male)
        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'Yes' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'Days_missed', mean_daysmissed_grad_yes_female)
        if pd.isnull(row['Days_missed']) and row['Graduated'] == 'No' and row['Gender'] == 'Female':
            df_double_conditional.set_value(index,'Days_missed', mean_daysmissed_grad_yes_male)

    df_double_conditional.to_csv('double_conditional.csv')
                                                        
    return None