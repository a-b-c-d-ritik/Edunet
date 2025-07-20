'''import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=(5,2),random_state=2,max_iter=2000)

modelobj=pickle.load(open('bestmodel.pkl','rb'))
data1=[25,3,226802,7,4,6,3,2,1,0,0,40,39]
data2=[25,'Private',226802,7,'Never-married','Machine-op-inspct','Own-child','Black','Male',0,0,40,'United-States']

scaled_data=modelobj.transform(data1)
result=clf.predict(scaled_data)

st.text(result)'''

import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Example flow
scaler = StandardScaler()
data = scaler.fit_transform(data)  # use transform here
model = MLPClassifier()
model.fit(data, labels)  # assuming labels is defined

# Later for predictions
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)




# Load preprocessors and model
#scaler = pickle.load(open('scaler.pkl', 'rb'))  # Only if you used a scaler during training
model = pickle.load(open('bestmodel.pkl', 'rb'))

# Example user input (should come from Streamlit widgets in production)
data = [[25, 3, 226802, 7, 4, 6, 3, 2, 1, 0, 0, 40, 39]]

# Apply scaling
scaled_data = model.transform(data)
result = model.predict(scaled_data)

st.text(f'Predicted label: {result[0]}')
