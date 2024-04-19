import DataPreProcessing as pre
data = {
     'id': 6,
    'age': 68.0,
    'bp': 70.0,
    'sg': 1.01,
    'al': 0.0,
    'su': 0.0,
    'rbc': None,
    'pc': 'normal',
    'pcc': 'notpresent',
    'ba': 'notpresent',
    'bgr': 100.0,
    'bu': 54.0,
    'sc': 24.0,
    'sod': 104.0,
    'pot': 4.0,
    'hemo': 12.4,
    'pcv': 36,
    'wc': None,
    'rc': None,
    'htn': 'no',
    'dm': 'no',
    'cad': 'no',
    'appet': 'good',
    'pe': 'no',
    'ane': 'no',
    'classification': 'ckd'
}

data1 = {key: [value] for key, value in data.items()}
# print(data1)
print(pre.preprocess(data1))
print(data1)
