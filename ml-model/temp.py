import model
import DataPreProcessing as dp
print(model.predict({
    'id': 0,
    'age': 48.0,
    'bp': 80.0,
    'sg': 1.02,
    'al': 1.0,
    'su': 0.0,
    'rbc': 'yes',  # Assuming this is an empty string
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': 121.0,
    'bu': 36.0,
    'sc': 1.2,
    'sod': 'yes',  # Assuming this is an empty string
    'pot': 'yes',  # Assuming this is an empty string
    'hemo': 15.4,
    'pcv': 44,
    'wc': 7800,
    'rc': 5.2,
    'htn': 'yes',
    'dm': 'yes',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no',
}))
