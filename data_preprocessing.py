def load_and_preprocess_data(csv_path="housing.csv",targetColumn = 'median_house_value', train_size=0.8, random_state=42,columnsToIgnore = []):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Charger le CSV
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns)

    #Chose target and ignore what user wants
    A = df.drop(columns=columnsToIgnore + [targetColumn])
    b = df[targetColumn].values

    #save this to go back to unnormalized values
    b_mean = b.mean()
    b_std = b.std()

    #replace categorical variables by booleans
    A = pd.get_dummies(A)

    # Normalization
    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    b = (b - b.mean()) / b.std()

    # Train/test split
    A_train, A_test, b_train, b_test = train_test_split(
        A, b, train_size=train_size, random_state=random_state
    )

    return A_train,b_train,b_mean, b_std




def load_and_preprocess_Housing_data(train_size=0.8, random_state=42):
    
    return load_and_preprocess_data(csv_path="dataset/housing.csv",targetColumn = 'median_house_value',train_size=train_size, random_state=random_state)

def load_and_preprocess_Student_Performance_data(train_size=0.8, random_state=42):
    
    return load_and_preprocess_data(csv_path="dataset/Student_Performance.csv",targetColumn = 'Performance Index',train_size=train_size, random_state=random_state)

def load_and_preprocess_Insurance_data(train_size=0.8, random_state=42):
    return load_and_preprocess_data(csv_path="dataset/insurance.csv",targetColumn = 'charges',train_size=train_size, random_state=random_state)

def load_and_preprocess_genes_data(train_size=0.8, random_state=42):
    return load_and_preprocess_data(csv_path="dataset/genes_combined.csv",targetColumn = 'target',train_size=train_size, random_state=random_state)